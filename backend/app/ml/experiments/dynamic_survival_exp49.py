"""Experiment 49: dynamic survival with right-censored active histories.

The challenger fits a penalized Cox time-varying model to monthly start/stop
intervals. Completed projects contribute one event; projects still active at a
training cutoff contribute a right-censored terminal interval. Current Exp32
AFT remaining-time is retained as a training-OOF-selected anchor, and the exact
Exp34 fallback remains untouched outside production's evidence-only AFT route.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _aft_remaining_prediction, _corrections, _delay_from_remaining,
    _fit_aft_family_models, _fit_residual_calibration, _public_calibration, _remaining_frame,
)
from backend.app.ml.experiments.path_oof_delay_exp34 import FAMILIES, _rolling_folds, enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger, write_experiment_prediction_ledger
from backend.app.ml.experiments.trajectory_exp12 import engineer_history
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows, enrich_supervised_for_production
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel, CALIBRATION_GATE_FEATURE, _select_aft_calibration_projects

EXPERIMENT_ID = "exp_49"
EXPERIMENT_SEQUENCE = 49
EXPERIMENT_NAME = "Dynamic censored survival for remaining completion time"
EXPERIMENT_SCOPE = "delay"
HYPOTHESIS = (
    "Time-varying monthly histories from both completed and right-censored active projects "
    "add duration information beyond completed-only Exp32 AFT regression."
)
CHANGED_DIMENSION = "censored_training_population_and_duration_model"
SURVIVAL_PENALIZER = 0.10
MAX_REMAINING_DAYS = 20.0 * 365.25
BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)  # weight on dynamic survival

SURVIVAL_FEATURE_CANDIDATES = [
    "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "expenditure_ratio",
    "cost_escalation_percentage", "schedule_slippage_days", "schedule_slippage_ratio",
    "elapsed_duration_days", "planned_duration_days", "duration_ratio", "expected_progress_percentage",
    "sector_average_delay", "sector_average_cost_overrun", "sector_delay_rate", "sector_cost_overrun_rate",
    "agency_average_delay", "agency_average_cost_overrun", "agency_delay_rate", "agency_cost_overrun_rate",
    "exp12_history_12m", "exp12_cost_velocity_12m", "exp12_cost_revisions_12m",
    "exp12_months_since_cost_revision", "exp12_cost_volatility_6m",
    "exp12_expenditure_velocity_3m", "exp12_expenditure_velocity_6m", "exp12_expenditure_velocity_12m",
    "exp12_expenditure_acceleration", "exp12_slippage_velocity_3m", "exp12_slippage_velocity_6m",
    "exp12_slippage_velocity_12m", "exp12_slippage_acceleration", "exp12_schedule_revisions_12m",
    "exp12_months_since_schedule_revision", "exp12_slippage_volatility_6m",
    "exp34_observations_seen", "exp34_months_observed", "exp34_cost_revision_count",
    "exp34_schedule_revision_count", "exp34_cumulative_abs_cost_revision_pct",
    "exp34_max_cost_escalation", "exp34_cost_recovery_from_peak", "exp34_max_schedule_slippage",
    "exp34_delay_recovery_from_peak", "exp34_slippage_positive_share", "exp34_cost_overrun_positive_share",
    "exp34_cost_worsening_share", "exp34_delay_worsening_share",
    "exp34_months_since_first_cost_revision", "exp34_months_since_first_schedule_revision",
]
FORBIDDEN_INPUTS = {"actual_completion_date", "actual_delay_days", "actual_cost_overrun_percentage", "reported_completion_expenditure_cr"}
SOURCE_DATE_COLUMNS = ["snapshot_date", "planned_start_date", "approval_date", "completion_date"]


def _canonical_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_project_id", "snapshot_date"}
    missing = sorted(required.difference(history.columns))
    if missing: raise ValueError("Exp49 history is missing: " + ", ".join(missing))
    frame = history.copy(); frame["canonical_project_id"] = frame["canonical_project_id"].astype("string").str.strip()
    for column in SOURCE_DATE_COLUMNS:
        if column not in frame: frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"])
    # Duplicate choice excludes completion/outcome fields so a future label can
    # never alter an earlier interval's covariates.
    safe_columns = sorted(column for column in frame.columns if column not in FORBIDDEN_INPUTS | {"completion_date", "completion_year", "actual_risk"})
    safe = frame.reindex(columns=safe_columns).astype("string").fillna("<NA>")
    frame["_exp49_tie"] = pd.util.hash_pandas_object(safe, index=False).to_numpy(np.uint64)
    frame = frame.sort_values(["canonical_project_id", "snapshot_date", "_exp49_tie"], kind="mergesort")
    return frame.drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last").drop(columns="_exp49_tie").reset_index(drop=True)


def _enrich_archive(history: pd.DataFrame) -> pd.DataFrame:
    canonical = _canonical_history(history)
    trajectory = engineer_history(canonical)
    return enrich_path_dependence(trajectory, history=trajectory)


def _project_origin(group: pd.DataFrame) -> pd.Timestamp:
    first = group.iloc[0]; first_snapshot = pd.Timestamp(first["snapshot_date"])
    for column in ("planned_start_date", "approval_date"):
        value = first.get(column)
        if pd.notna(value) and pd.Timestamp(value) <= first_snapshot:
            return pd.Timestamp(value)
    return first_snapshot


def build_survival_risk_set(
    history: pd.DataFrame,
    *,
    cutoff: pd.Timestamp | str,
    excluded_project_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Construct ordered time-varying intervals with one event at most/project."""
    cutoff = pd.Timestamp(cutoff)
    frame = _enrich_archive(history)
    frame = frame[frame.snapshot_date.le(cutoff)].copy()
    excluded = set(str(value) for value in (excluded_project_ids or set()))
    frame = frame[~frame.canonical_project_id.astype("string").isin(excluded)]
    intervals: list[dict[str, object]] = []
    for project_id, group in frame.groupby("canonical_project_id", sort=False):
        group = group.sort_values("snapshot_date", kind="mergesort")
        origin = _project_origin(group)
        known = pd.to_datetime(group["completion_date"], errors="coerce").dropna()
        # A later completion date is never used as an event/duration at this
        # cutoff. Such a project is censored exactly like an outcome-unknown one.
        eligible_completion = known[known.le(cutoff)]
        completion = eligible_completion.min() if len(eligible_completion) else pd.NaT
        reports = group[group.snapshot_date.lt(completion if pd.notna(completion) else cutoff)].copy()
        if reports.empty: continue
        report_rows = list(reports.iterrows())
        for position, (_, row) in enumerate(report_rows):
            start = max(0.0, (pd.Timestamp(row.snapshot_date) - origin).total_seconds() / 86400.0)
            boundary = pd.Timestamp(report_rows[position + 1][1].snapshot_date) if position + 1 < len(report_rows) else cutoff
            event = False
            if pd.notna(completion) and completion <= boundary:
                boundary = pd.Timestamp(completion); event = True
            stop = (boundary - origin).total_seconds() / 86400.0
            if stop <= start: continue
            item = row.to_dict(); item.update({
                "canonical_project_id": str(project_id), "exp49_origin_date": origin,
                "exp49_start_days": float(start), "exp49_stop_days": float(stop),
                "exp49_event": int(event), "exp49_remaining_observed_days": float(stop - start),
            })
            intervals.append(item)
            if event: break
    risk = pd.DataFrame(intervals)
    if risk.empty: raise ValueError("Exp49 survival risk set is empty")
    if (risk.exp49_stop_days <= risk.exp49_start_days).any(): raise AssertionError("Exp49 intervals must have positive ordered duration")
    if (risk.groupby("canonical_project_id").exp49_event.sum() > 1).any(): raise AssertionError("Exp49 allows at most one completion event per project")
    risk = assign_project_balanced_weights(risk)
    return risk


def _select_features(risk: pd.DataFrame) -> list[str]:
    selected = []
    for name in SURVIVAL_FEATURE_CANDIDATES:
        if name not in risk: continue
        values = pd.to_numeric(risk[name], errors="coerce")
        if values.notna().mean() >= .05 and values.dropna().nunique() > 1:
            selected.append(name)
    if len(selected) < 5: raise ValueError(f"Exp49 has too few usable survival covariates: {selected}")
    return selected


def _fit_transform_contract(risk: pd.DataFrame, features: list[str]) -> dict:
    contract = {}
    for name in features:
        values = pd.to_numeric(risk[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median()) if values.notna().any() else 0.0
        finite = values.dropna().to_numpy(float)
        q1, q3 = np.quantile(finite, [.25, .75]) if len(finite) else (0.0, 1.0)
        scale = max(float(q3 - q1), float(np.std(finite)) if len(finite) else 1.0, 1e-6)
        contract[name] = {"median": median, "scale": scale, "missing_indicator": bool(values.isna().any())}
    return contract


def _transform(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for name, spec in contract.items():
        values = pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        result[name] = np.clip((values.fillna(spec["median"]) - spec["median"]) / spec["scale"], -8.0, 8.0)
        if spec["missing_indicator"]: result[f"{name}__missing"] = values.isna().astype(float)
    return result


def fit_dynamic_survival(risk: pd.DataFrame) -> dict:
    features = _select_features(risk); contract = _fit_transform_contract(risk, features); design = _transform(risk, contract)
    design.insert(0, "canonical_project_id", risk.canonical_project_id.astype(str).to_numpy())
    design["start"] = risk.exp49_start_days.to_numpy(float); design["stop"] = risk.exp49_stop_days.to_numpy(float)
    design["event"] = risk.exp49_event.to_numpy(int); design["weight"] = risk.sample_weight.to_numpy(float)
    model = CoxTimeVaryingFitter(penalizer=SURVIVAL_PENALIZER, l1_ratio=0.0)
    model.fit(design, id_col="canonical_project_id", start_col="start", stop_col="stop", event_col="event", weights_col="weight", robust=False, show_progress=False)
    final = risk.sort_values("exp49_stop_days").groupby("canonical_project_id", sort=False).tail(1)
    final_design = _transform(final, contract)
    risk_score = model.predict_partial_hazard(final_design).to_numpy(float)
    cindex = concordance_index(final.exp49_stop_days.to_numpy(float), -risk_score, final.exp49_event.to_numpy(int)) if final.exp49_event.sum() and len(final) > 2 else None
    return {"model": model, "transform": contract, "features": features, "concordance_index": float(cindex) if cindex is not None and np.isfinite(cindex) else None}


def _baseline_at(times: np.ndarray, cumulative: np.ndarray, query: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(times, query, side="right") - 1
    result = np.zeros(len(query), dtype=float); valid = positions >= 0; result[valid] = cumulative[positions[valid]]
    return result


def predict_conditional_median_remaining(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Freeze current covariates and invert conditional Cox survival at 0.5."""
    model = bundle["model"]; design = _transform(frame, bundle["transform"])
    partial = np.clip(model.predict_partial_hazard(design).to_numpy(float), 1e-8, 1e8)
    age = pd.to_numeric(frame["exp49_current_age_days"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    baseline = model.baseline_cumulative_hazard_.iloc[:, 0]
    times = baseline.index.to_numpy(float); cumulative = baseline.to_numpy(float)
    current_h = _baseline_at(times, cumulative, age); target_h = current_h + math.log(2.0) / partial
    prediction = np.empty(len(frame), dtype=float)
    tail_rate = max((cumulative[-1] - cumulative[-2]) / max(times[-1] - times[-2], 1e-6), 1e-6) if len(times) >= 2 else 1e-6
    for i, target in enumerate(target_h):
        position = int(np.searchsorted(cumulative, target, side="left"))
        completion_age = times[position] if position < len(times) else times[-1] + (target - cumulative[-1]) / tail_rate
        prediction[i] = np.clip(completion_age - age[i], 0.0, MAX_REMAINING_DAYS)
    return prediction


def _attach_origins(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    canonical = _canonical_history(history); origins = []
    for project_id, group in canonical.groupby("canonical_project_id", sort=False):
        origins.append({"canonical_project_id": str(project_id), "exp49_origin_date": _project_origin(group.sort_values("snapshot_date"))})
    lookup = pd.DataFrame(origins); rows = frame.copy(); rows["canonical_project_id"] = rows.canonical_project_id.astype("string")
    rows = rows.merge(lookup, on="canonical_project_id", how="left", validate="many_to_one")
    rows["exp49_origin_date"] = pd.to_datetime(rows.exp49_origin_date, errors="coerce").fillna(pd.to_datetime(rows.snapshot_date, errors="coerce"))
    rows["exp49_current_age_days"] = (pd.to_datetime(rows.snapshot_date) - rows.exp49_origin_date).dt.total_seconds().div(86400).clip(lower=0)
    return rows


def _delay_weights(production_model) -> dict[str, float]:
    weights = {family: float(production_model.weights.get(family, 0.0)) for family in FAMILIES}
    if abs(sum(weights.values()) - 1.0) > 1e-9: raise ValueError(f"Exp49 requires normalized production AFT weights: {weights}")
    return weights


def training_oof_survival_blend(
    train: pd.DataFrame, history: pd.DataFrame, delay_features: list[str], delay_weights: dict[str, float],
    *, holdout_ids: set[str],
) -> tuple[float, dict, dict]:
    chunks = [];
    for fitting, validation, year in _rolling_folds(train):
        validation = _remaining_frame(validation)
        if validation.empty: continue
        excluded = set(validation.canonical_project_id.astype(str)) | holdout_ids
        risk = build_survival_risk_set(history, cutoff=pd.Timestamp(year=year - 1, month=12, day=31), excluded_project_ids=excluded)
        survival = fit_dynamic_survival(risk)
        aft_models = _fit_aft_family_models(_remaining_frame(fitting), delay_features)
        current_remaining = _aft_remaining_prediction(aft_models, delay_weights, validation, delay_features)
        survival_remaining = predict_conditional_median_remaining(survival, validation)
        chunk = validation[["actual_delay_days", "sample_weight", "canonical_project_id", "lifecycle_stage", "snapshot_date", "planned_completion_date"]].copy()
        chunk["aft_remaining"] = current_remaining; chunk["survival_remaining"] = survival_remaining; chunk["fold_year"] = year
        chunks.append(chunk)
    if len(chunks) < 2: raise ValueError("Exp49 requires at least two rolling OOF survival folds")
    oof = pd.concat(chunks, ignore_index=True)
    comparisons = []
    for weight in BLEND_GRID:
        remaining = (1.0 - weight) * oof.aft_remaining.to_numpy(float) + weight * oof.survival_remaining.to_numpy(float)
        prediction = _delay_from_remaining(oof, remaining)
        mae = float(mean_absolute_error(oof.actual_delay_days, prediction, sample_weight=oof.sample_weight))
        comparisons.append({"survival_weight": weight, "MAE": mae})
    best = min(comparisons, key=lambda item: (item["MAE"], item["survival_weight"]))
    weight = float(best["survival_weight"])
    remaining = (1.0 - weight) * oof.aft_remaining.to_numpy(float) + weight * oof.survival_remaining.to_numpy(float)
    oof["prediction"] = _delay_from_remaining(oof, remaining); oof["residual"] = oof.actual_delay_days - oof.prediction
    calibration = _fit_residual_calibration(oof)
    return weight, calibration, {"folds": sorted(oof.fold_year.unique().tolist()), "oof_rows": int(len(oof)), "oof_projects": int(oof.canonical_project_id.nunique()), "blend_grid": comparisons, "selected_survival_weight": weight}


def _route_metrics(rows: pd.DataFrame, mask: np.ndarray) -> dict:
    part = rows.loc[mask]
    if part.empty: return {"projects": 0, "snapshots": 0, "production_delay_mae": None, "experiment_delay_mae": None}
    return {"projects": int(part.canonical_project_id.nunique()), "snapshots": int(len(part)), "production_delay_mae": _regression_metrics(part.actual_delay_days, part.production_delay_prediction.to_numpy(float), part.sample_weight, part.canonical_project_id)["MAE"], "experiment_delay_mae": _regression_metrics(part.actual_delay_days, part.experiment_delay_prediction.to_numpy(float), part.sample_weight, part.canonical_project_id)["MAE"]}


def _write_artifacts(directory: Path, *, survival_audit: dict, bootstrap: dict, config: dict, comparison: dict) -> dict:
    payloads = {"risk_set_audit.json": survival_audit, "survival_model_configuration.json": config, "feature_target_audit.json": {"changed_dimension": CHANGED_DIMENSION, "forbidden_inputs": sorted(FORBIDDEN_INPUTS), "point_forecast": "conditional median remaining time", "later_completion_after_cutoff_is_censored": True}, "bootstrap_results.json": bootstrap}
    for name, payload in payloads.items(): (directory / name).write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    (directory / "experiment_summary.md").write_text(f"# Experiment 49 comparison\n\n{HYPOTHESIS}\n\nProduction Delay MAE: {comparison['production_delay_mae']}\n\nExperiment Delay MAE: {comparison['experiment_delay_mae']}\n\nImprovement: {comparison['delay_improvement_percentage']}%\n\nExecution verdict: EXECUTION VALID\n\nScientific verdict: {comparison['scientific_verdict']}\n")
    return {name: str(directory / name) for name in [*payloads, "experiment_summary.md"]}


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None, **_):
    if history is None:
        if not TRAJECTORIES.exists(): raise FileNotFoundError("Exp49 requires the full monthly trajectory archive")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    base = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    base["completion_year"] = pd.to_numeric(base.completion_year, errors="coerce"); base["snapshot_date"] = pd.to_datetime(base.snapshot_date, errors="coerce")
    base = _attach_origins(base, history); train, test = temporal_project_split(base, training_start, training_end, test_end)
    if set(train.canonical_project_id) & set(test.canonical_project_id): raise ValueError("Exp49 train/holdout project overlap")
    production_delay_model = production_bundle["delay"]
    if not isinstance(production_delay_model, AFTResidualDelayModel): raise TypeError("Exp49 baseline must be current AFTResidualDelayModel production")
    delay_features = list(production_delay_model.features); delay_weights = _delay_weights(production_delay_model)
    holdout_ids = set(test.canonical_project_id.astype(str))
    blend_weight, calibration, oof = training_oof_survival_blend(train, history, delay_features, delay_weights, holdout_ids=holdout_ids)
    final_risk = build_survival_risk_set(history, cutoff=pd.Timestamp(year=training_end, month=12, day=31), excluded_project_ids=holdout_ids)
    if set(final_risk.canonical_project_id.astype(str)) & holdout_ids: raise AssertionError("Exp49 holdout project entered survival training")
    survival = fit_dynamic_survival(final_risk)

    compare = _production_cost_evaluation_rows(test); gate = _select_aft_calibration_projects(compare); compare = compare.copy()
    compare[CALIBRATION_GATE_FEATURE] = compare.canonical_project_id.astype("string").isin(gate); compare = assign_project_balanced_weights(compare)
    production_delay = np.maximum(0.0, production_delay_model.predict(compare)); candidate_delay = production_delay.copy()
    eligible = AFTResidualDelayModel._aft_eligible(compare).to_numpy(bool)
    subset = compare.loc[eligible].copy()
    aft_remaining = _aft_remaining_prediction(production_delay_model.aft_models, production_delay_model.weights, subset, production_delay_model.features)
    survival_remaining = predict_conditional_median_remaining(survival, subset)
    blended_remaining = (1.0 - blend_weight) * aft_remaining + blend_weight * survival_remaining
    raw_delay = _delay_from_remaining(subset, blended_remaining)
    candidate_delay[np.flatnonzero(eligible)] = np.maximum(0.0, raw_delay + _corrections(subset, raw_delay, calibration))

    production_cost = production_bundle["cost"].predict(compare); candidate_cost = production_cost.copy()
    if not np.array_equal(production_cost, candidate_cost): raise AssertionError("Exp49 changed Cost predictions")
    prod = _regression_metrics(compare.actual_delay_days, production_delay, compare.sample_weight, compare.canonical_project_id)
    exp = _regression_metrics(compare.actual_delay_days, candidate_delay, compare.sample_weight, compare.canonical_project_id)
    cost = _regression_metrics(compare.actual_cost_overrun_percentage, production_cost, compare.sample_weight, compare.canonical_project_id)
    absolute = float(prod["MAE"]) - float(exp["MAE"]); percentage = absolute / float(prod["MAE"]) * 100.0
    scored = compare.copy(); scored["production_delay_prediction"] = production_delay; scored["experiment_delay_prediction"] = candidate_delay
    scored["production_cost_prediction"] = production_cost; scored["experiment_cost_prediction"] = candidate_cost
    scored["experiment_route"] = np.where(eligible, "exp49_survival_aft_blend", "exp34_fallback")
    bootstrap = paired_project_mae_comparison(scored, actual="actual_delay_days", baseline_prediction="production_delay_prediction", candidate_prediction="experiment_delay_prediction", bootstrap_samples=5000, seed=49000 + int(training_end))
    verdict = "PROMOTION CANDIDATE" if percentage > 0 and bootstrap["probability_candidate_better"] >= .5 else "DO NOT PROMOTE"
    eligible_metrics = _route_metrics(scored, eligible); fallback_metrics = _route_metrics(scored, ~eligible)
    if fallback_metrics["production_delay_mae"] != fallback_metrics["experiment_delay_mae"]: raise AssertionError("Exp49 changed fallback predictions")
    survival_only_delay = _delay_from_remaining(subset, survival_remaining)
    survival_only_metrics = _regression_metrics(subset.actual_delay_days, survival_only_delay, subset.sample_weight, subset.canonical_project_id)
    comparison = {"production_delay_mae": prod["MAE"], "experiment_delay_mae": exp["MAE"], "absolute_delay_mae_improvement": round(absolute, 6), "delay_improvement_percentage": round(percentage, 6), "production_cost_mae": cost["MAE"], "experiment_cost_mae": cost["MAE"], "cost_predictions_identical": True, "comparison_test_projects": int(compare.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(compare)), "survival_projects": int(subset.canonical_project_id.nunique()), "fallback_projects": int(compare.loc[~eligible].canonical_project_id.nunique()), "survival_snapshots": int(eligible.sum()), "fallback_snapshots": int((~eligible).sum()), "survival_route": eligible_metrics, "fallback_route": fallback_metrics, "survival_alone_route_mae": survival_only_metrics["MAE"], "paired_project_bootstrap": bootstrap, "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict}
    event_projects = int(final_risk.groupby("canonical_project_id").exp49_event.max().sum()); censored_projects = int(final_risk.canonical_project_id.nunique() - event_projects)
    risk_audit = {"intervals": int(len(final_risk)), "unique_projects": int(final_risk.canonical_project_id.nunique()), "completed_event_projects": event_projects, "censored_training_projects": censored_projects, "event_censoring_ratio": event_projects / max(censored_projects, 1), "events_at_most_once_per_project": bool((final_risk.groupby("canonical_project_id").exp49_event.sum() <= 1).all()), "positive_ordered_intervals": bool((final_risk.exp49_stop_days > final_risk.exp49_start_days).all()), "training_cutoff": f"{training_end}-12-31", "holdout_projects_excluded": len(holdout_ids), "concordance_index": survival["concordance_index"]}
    config = {"model": "lifelines.CoxTimeVaryingFitter", "penalizer": SURVIVAL_PENALIZER, "point_forecast": "conditional median with frozen current covariates", "max_remaining_days": MAX_REMAINING_DAYS, "selected_features": survival["features"], "transform": survival["transform"], "blend": oof, "calibration": _public_calibration(calibration)}

    run_id = f"exp49-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"; window = f"{training_start}_{training_end}"
    ledger = build_prediction_ledger(scored, experiment_id=EXPERIMENT_ID, window=window, production_delay_prediction=production_delay, experiment_delay_prediction=candidate_delay, extra_columns=["completion_year", "lifecycle_stage", "sector", "implementing_agency", "state", "project_size_category", "approved_cost_cr", "schedule_slippage_days", "duration_ratio", "exp12_history_12m", "exp34_observations_seen", "experiment_route"])
    assert_prediction_ledger_matches_cohort(ledger, compare)
    persisted = write_experiment_prediction_ledger(ledger, experiment_id=EXPERIMENT_ID, window=window, run_id=run_id, extra_manifest={"primary_target": "delay", "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict, "changed_dimension": CHANGED_DIMENSION, "bootstrap_samples": 5000, "cost_unchanged": True, "survival_projects": int(subset.canonical_project_id.nunique()), "fallback_projects": int(compare.loc[~eligible].canonical_project_id.nunique()), "censored_training_projects": censored_projects})
    artifacts = _write_artifacts(Path(persisted["ledger_path"]).parent, survival_audit=risk_audit, bootstrap=bootstrap, config=config, comparison=comparison)
    lookup = {(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): float(pred) for (_, row), pred in zip(scored.iterrows(), candidate_delay)}
    return {"experiment": {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE, "run_id": run_id, "model_role": "experiment", "promotion_allowed": False, "changed_dimension": CHANGED_DIMENSION, "hypothesis": HYPOTHESIS, "survival_model": "penalized Cox time-varying", "selected_survival_weight": blend_weight, "rolling_oof": oof, "calibration": _public_calibration(calibration), "risk_set_audit": risk_audit, "future_holdout_used_for_training_or_selection": False, "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict, "ledger_path": str(persisted["ledger_path"]), "ledger_manifest_path": str(persisted["manifest_path"]), "cohort_fingerprint": persisted["manifest"]["cohort_fingerprint"], "ledger_fingerprint": persisted["manifest"]["ledger_fingerprint"], "audit_artifacts": artifacts}, "overall_comparison": comparison, "state": {"lookup": lookup}}


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    keys = set(state.get("lookup", {})); mask = [(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) in keys for _, row in frame.iterrows()]
    return assign_project_balanced_weights(frame.loc[mask].copy())


def predict_project(row: pd.Series, state: dict) -> dict:
    key = (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
    if key not in state.get("lookup", {}): raise ValueError("Exp49 row is outside the frozen comparison cohort")
    return {"delay_days": float(state["lookup"][key])}
