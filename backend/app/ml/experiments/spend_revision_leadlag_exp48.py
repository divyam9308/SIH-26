"""Experiment 48: leakage-safe spending-to-Cost-revision lead/lag coupling.

This feature-only challenger retains production's Cost target, selected model
family, seed, cohort, project weights and Exp33 rolling-OOF calibration method.
Every coupling statistic is computed from the current report prefix only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import _corrections, _cost_calibration_oof, _public_calibration
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger, write_experiment_prediction_ledger
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _json_safe, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED, _production_cost_evaluation_rows, enrich_supervised_for_production
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE, _select_aft_calibration_projects

EXPERIMENT_ID = "exp_48"
EXPERIMENT_SEQUENCE = 48
EXPERIMENT_NAME = "Spending-to-Cost-revision lead-lag coupling"
EXPERIMENT_SCOPE = "cost"
HYPOTHESIS = (
    "Changes in financial execution systematically lead or lag formal revised-Cost changes, "
    "revealing Cost pressure before the official revised-Cost field catches up."
)
CHANGED_DIMENSION = "feature_set"
MIN_REVISION_PP = 0.25
SPEND_REGIME_VELOCITY_PP_PER_MONTH = 0.25
LAGS_MONTHS = (1, 3, 6)

EXP48_FEATURES = [
    "exp48_spend_approved_velocity_3m", "exp48_spend_approved_velocity_6m", "exp48_spend_approved_velocity_12m",
    "exp48_spend_revised_velocity_3m", "exp48_spend_revised_velocity_6m", "exp48_spend_revised_velocity_12m",
    "exp48_spend_approved_acceleration", "exp48_spend_recent_vs_previous_3m",
    "exp48_months_acceleration_to_revision", "exp48_acceleration_precedes_latest_revision",
    "exp48_slowdown_precedes_latest_upward_revision", "exp48_spend_pp_since_cost_revision",
    "exp48_escalation_pp_since_spend_regime_break", "exp48_months_execution_since_cost_revision",
    "exp48_execution_pressure_since_revision", "exp48_financial_vs_elapsed_gap_pp",
    "exp48_spend_escalation_divergence_3m", "exp48_spend_escalation_divergence_6m",
    "exp48_spend_leads_cost_corr_lag1m", "exp48_spend_leads_cost_corr_lag3m", "exp48_spend_leads_cost_corr_lag6m",
    "exp48_cost_leads_spend_corr_lag1m", "exp48_cost_leads_spend_corr_lag3m", "exp48_cost_leads_spend_corr_lag6m",
    "exp48_strongest_leadlag_abs_correlation", "exp48_strongest_leadlag_months", "exp48_strongest_leadlag_direction",
]
SOURCE_COLUMNS = {
    "canonical_project_id", "snapshot_date", "approved_cost_cr", "revised_cost_cr",
    "cumulative_expenditure_cr", "duration_ratio",
}
FORBIDDEN_INPUTS = {
    "completion_date", "actual_completion_date", "actual_cost_overrun_percentage",
    "reported_completion_expenditure_cr", "actual_delay_days",
}


def _canonical_history(history: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(SOURCE_COLUMNS.difference(history.columns))
    if missing:
        raise ValueError("Exp48 history is missing: " + ", ".join(missing))
    frame = history.copy()
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string").str.strip()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    for column in ("approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "duration_ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"])
    safe = frame.reindex(columns=sorted(SOURCE_COLUMNS)).astype("string").fillna("<NA>")
    frame["_exp48_tie"] = pd.util.hash_pandas_object(safe, index=False).to_numpy(np.uint64)
    frame = frame.sort_values(["canonical_project_id", "snapshot_date", "_exp48_tie"], kind="mergesort")
    return frame.drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last").drop(columns="_exp48_tie").reset_index(drop=True)


def _slope(dates: list[pd.Timestamp], values: list[float], current: pd.Timestamp, *, start_days: int, end_days: int = 0) -> float:
    positions = [i for i, date in enumerate(dates) if end_days <= (current - date).days <= start_days and np.isfinite(values[i])]
    if len(positions) < 2:
        return 0.0
    first = dates[positions[0]]
    x = np.asarray([(dates[i] - first).days / 30.4375 for i in positions], dtype=float)
    y = np.asarray([values[i] for i in positions], dtype=float)
    if np.ptp(x) <= 1e-12:
        return 0.0
    x = x - x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def _lagged_value(dates: list[pd.Timestamp], values: list[float], current_position: int, lag_months: int) -> float:
    target = dates[current_position] - pd.Timedelta(days=round(lag_months * 30.4375))
    candidates = [i for i in range(current_position) if dates[i] <= target and np.isfinite(values[i])]
    return values[candidates[-1]] if candidates else math.nan


def _safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])
    return value if np.isfinite(value) else 0.0


def engineer_spend_revision_leadlag(history: pd.DataFrame) -> pd.DataFrame:
    """Engineer compact lead/lag features using only each report's prefix."""
    frame = _canonical_history(history)
    records: list[dict[str, object]] = []
    for _, group in frame.groupby("canonical_project_id", sort=False):
        dates: list[pd.Timestamp] = []
        spend_approved: list[float] = []
        spend_revised: list[float] = []
        escalation: list[float] = []
        spend_delta: list[float] = []
        escalation_delta: list[float] = []
        last_revision_position = -1
        last_revision_positive = False
        last_acceleration_position = -1
        last_slowdown_position = -1
        last_spend_break_position = -1
        lead_pairs = {lag: ([], []) for lag in LAGS_MONTHS}
        reverse_pairs = {lag: ([], []) for lag in LAGS_MONTHS}

        for position, (_, row) in enumerate(group.iterrows()):
            current = pd.Timestamp(row["snapshot_date"])
            dates.append(current)
            approved = float(row["approved_cost_cr"]) if pd.notna(row["approved_cost_cr"]) else math.nan
            revised = float(row["revised_cost_cr"]) if pd.notna(row["revised_cost_cr"]) else math.nan
            spend = float(row["cumulative_expenditure_cr"]) if pd.notna(row["cumulative_expenditure_cr"]) else math.nan
            sa = spend / approved * 100.0 if np.isfinite(spend) and np.isfinite(approved) and approved > 0 else math.nan
            sr = spend / revised * 100.0 if np.isfinite(spend) and np.isfinite(revised) and revised > 0 else math.nan
            ce = (revised - approved) / approved * 100.0 if np.isfinite(revised) and np.isfinite(approved) and approved > 0 else math.nan
            spend_approved.append(sa)
            spend_revised.append(sr)
            escalation.append(ce)
            ds = sa - spend_approved[-2] if position and np.isfinite(sa) and np.isfinite(spend_approved[-2]) else math.nan
            dc = ce - escalation[-2] if position and np.isfinite(ce) and np.isfinite(escalation[-2]) else math.nan
            spend_delta.append(ds)
            escalation_delta.append(dc)

            v3 = _slope(dates, spend_approved, current, start_days=92)
            v6 = _slope(dates, spend_approved, current, start_days=183)
            v12 = _slope(dates, spend_approved, current, start_days=366)
            rv3 = _slope(dates, spend_revised, current, start_days=92)
            rv6 = _slope(dates, spend_revised, current, start_days=183)
            rv12 = _slope(dates, spend_revised, current, start_days=366)
            previous3 = _slope(dates, spend_approved, current, start_days=183, end_days=93)
            acceleration = v3 - v6
            if acceleration >= SPEND_REGIME_VELOCITY_PP_PER_MONTH:
                last_acceleration_position = position
                last_spend_break_position = position
            elif acceleration <= -SPEND_REGIME_VELOCITY_PP_PER_MONTH:
                last_slowdown_position = position
                last_spend_break_position = position
            revision_now = np.isfinite(dc) and abs(dc) >= MIN_REVISION_PP
            if revision_now:
                last_revision_position = position
                last_revision_positive = dc > 0

            for lag in LAGS_MONTHS:
                prior_spend = _lagged_value(dates, spend_delta, position, lag)
                prior_cost = _lagged_value(dates, escalation_delta, position, lag)
                if np.isfinite(dc) and np.isfinite(prior_spend):
                    lead_pairs[lag][0].append(float(prior_spend)); lead_pairs[lag][1].append(float(dc))
                if np.isfinite(ds) and np.isfinite(prior_cost):
                    reverse_pairs[lag][0].append(float(prior_cost)); reverse_pairs[lag][1].append(float(ds))

            record: dict[str, object] = {
                "canonical_project_id": row["canonical_project_id"], "snapshot_date": current,
                **{feature: 0.0 for feature in EXP48_FEATURES},
                "exp48_spend_approved_velocity_3m": v3, "exp48_spend_approved_velocity_6m": v6,
                "exp48_spend_approved_velocity_12m": v12, "exp48_spend_revised_velocity_3m": rv3,
                "exp48_spend_revised_velocity_6m": rv6, "exp48_spend_revised_velocity_12m": rv12,
                "exp48_spend_approved_acceleration": acceleration,
                "exp48_spend_recent_vs_previous_3m": v3 - previous3,
                "exp48_financial_vs_elapsed_gap_pp": sa - float(row["duration_ratio"]) * 100.0 if np.isfinite(sa) and pd.notna(row["duration_ratio"]) else 0.0,
                "exp48_spend_escalation_divergence_3m": v3 - _slope(dates, escalation, current, start_days=92),
                "exp48_spend_escalation_divergence_6m": v6 - _slope(dates, escalation, current, start_days=183),
            }
            if last_revision_position >= 0:
                months = (current - dates[last_revision_position]).days / 30.4375
                record["exp48_months_execution_since_cost_revision"] = months
                record["exp48_execution_pressure_since_revision"] = v3 * months
                baseline = spend_approved[last_revision_position]
                record["exp48_spend_pp_since_cost_revision"] = sa - baseline if np.isfinite(sa) and np.isfinite(baseline) else 0.0
            if last_spend_break_position >= 0:
                baseline = escalation[last_spend_break_position]
                record["exp48_escalation_pp_since_spend_regime_break"] = ce - baseline if np.isfinite(ce) and np.isfinite(baseline) else 0.0
            if last_revision_position >= 0 and last_acceleration_position >= 0:
                record["exp48_months_acceleration_to_revision"] = (dates[last_revision_position] - dates[last_acceleration_position]).days / 30.4375
                record["exp48_acceleration_precedes_latest_revision"] = float(last_acceleration_position <= last_revision_position)
            if last_revision_positive and last_slowdown_position >= 0:
                record["exp48_slowdown_precedes_latest_upward_revision"] = float(last_slowdown_position <= last_revision_position)

            candidates: list[tuple[float, int, float]] = []
            for lag in LAGS_MONTHS:
                lead = _safe_corr(*lead_pairs[lag])
                reverse = _safe_corr(*reverse_pairs[lag])
                record[f"exp48_spend_leads_cost_corr_lag{lag}m"] = lead
                record[f"exp48_cost_leads_spend_corr_lag{lag}m"] = reverse
                candidates.extend([(abs(lead), lag, float(np.sign(lead))), (abs(reverse), -lag, float(np.sign(reverse)))])
            strongest = max(candidates, key=lambda item: (item[0], -abs(item[1])))
            record["exp48_strongest_leadlag_abs_correlation"] = strongest[0]
            record["exp48_strongest_leadlag_months"] = strongest[1]
            record["exp48_strongest_leadlag_direction"] = strongest[2]
            records.append(record)
    return pd.DataFrame.from_records(records, columns=["canonical_project_id", "snapshot_date", *EXP48_FEATURES])


def enrich_spend_revision_leadlag(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False) if TRAJECTORIES.exists() else supervised.copy()
    engineered = engineer_spend_revision_leadlag(history)
    rows = supervised.copy()
    rows["canonical_project_id"] = rows["canonical_project_id"].astype("string")
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    result = rows.merge(engineered, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")
    if len(result) != len(rows):
        raise AssertionError("Exp48 feature engineering changed the supervised cohort")
    for feature in EXP48_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(0.0)
    return result


def _cost_family(production_model) -> str:
    name = production_model.model.named_steps["model"].__class__.__name__.lower()
    if "extratrees" in name: return "extra_trees"
    if "lgbm" in name: return "lightgbm"
    if "xgb" in name: return "xgboost"
    raise ValueError(f"Unsupported production Cost family: {name}")


def _diagnostics(rows: pd.DataFrame) -> dict:
    work = rows.copy()
    work["_production"] = (work.production_cost_prediction - work.actual_cost_overrun_percentage).abs()
    work["_experiment"] = (work.experiment_cost_prediction - work.actual_cost_overrun_percentage).abs()
    per_project = work.groupby("canonical_project_id").agg(production=("_production", "mean"), experiment=("_experiment", "mean"))
    stages = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = work[work.get("lifecycle_stage", pd.Series(index=work.index, dtype="string")).eq(stage)]
        stages[stage] = {"projects": int(part.canonical_project_id.nunique()), "snapshots": int(len(part)), "production_mae": float(np.average(part._production, weights=part.sample_weight)) if len(part) else None, "experiment_mae": float(np.average(part._experiment, weights=part.sample_weight)) if len(part) else None}
    return _json_safe({
        "median_per_project_mae": {"production": per_project.production.median(), "experiment": per_project.experiment.median()},
        "p90_per_project_mae": {"production": per_project.production.quantile(.9), "experiment": per_project.experiment.quantile(.9)},
        "absolute_error_p90": {"production": work._production.quantile(.9), "experiment": work._experiment.quantile(.9)},
        "lifecycle_mae": stages,
    })


def _write_artifacts(directory: Path, *, features: list[str], bootstrap: dict, comparison: dict) -> dict:
    payloads = {
        "feature_target_audit.json": {"changed_dimension": CHANGED_DIMENSION, "source_columns": sorted(SOURCE_COLUMNS), "forbidden_inputs": sorted(FORBIDDEN_INPUTS), "past_prefix_only": True, "forward_auxiliary_dependency": False},
        "model_feature_config.json": {"new_features": EXP48_FEATURES, "final_features": features, "lags_months": LAGS_MONTHS, "revision_threshold_pp": MIN_REVISION_PP, "spend_regime_threshold_pp_per_month": SPEND_REGIME_VELOCITY_PP_PER_MONTH},
        "bootstrap_results.json": bootstrap,
    }
    for name, payload in payloads.items():
        (directory / name).write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    (directory / "experiment_summary.md").write_text(
        f"# Experiment 48 comparison\n\n{HYPOTHESIS}\n\nProduction Cost MAE: {comparison['production_cost_mae']}\n\nExperiment Cost MAE: {comparison['experiment_cost_mae']}\n\nImprovement: {comparison['cost_improvement_percentage']}%\n\nExecution verdict: EXECUTION VALID\n\nScientific verdict: {comparison['scientific_verdict']}\n"
    )
    return {name: str(directory / name) for name in [*payloads, "experiment_summary.md"]}


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None, **_):
    enriched = enrich_spend_revision_leadlag(enrich_path_dependence(enrich_supervised_for_production(data.copy())), history=history)
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    if set(train.canonical_project_id) & set(test.canonical_project_id):
        raise ValueError("Exp48 train/holdout project overlap")
    production_cost_model = production_bundle["cost"]
    base_features = list(production_cost_model.features)
    if set(base_features) & set(EXP48_FEATURES):
        raise ValueError("Exp48 duplicates a production feature")
    features = list(dict.fromkeys(base_features + EXP48_FEATURES))
    family = _cost_family(production_cost_model)
    calibration, oof = _cost_calibration_oof(train, features, family)
    candidate_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family], train, features, "actual_cost_overrun_percentage")

    compare = _production_cost_evaluation_rows(test)
    gate_projects = _select_aft_calibration_projects(compare)
    compare = compare.copy()
    compare[CALIBRATION_GATE_FEATURE] = compare.canonical_project_id.astype("string").isin(gate_projects)
    compare = assign_project_balanced_weights(compare)
    production_cost = production_cost_model.predict(compare)
    raw = candidate_model.predict(compare[features])
    candidate_cost = raw + _corrections(compare, raw, calibration)
    production_delay = np.maximum(0.0, production_bundle["delay"].predict(compare))
    candidate_delay = production_delay.copy()
    if not np.array_equal(production_delay, candidate_delay):
        raise AssertionError("Exp48 changed Delay predictions")
    prod_metrics = _regression_metrics(compare.actual_cost_overrun_percentage, production_cost, compare.sample_weight, compare.canonical_project_id)
    exp_metrics = _regression_metrics(compare.actual_cost_overrun_percentage, candidate_cost, compare.sample_weight, compare.canonical_project_id)
    delay_metrics = _regression_metrics(compare.actual_delay_days, production_delay, compare.sample_weight, compare.canonical_project_id)
    absolute = float(prod_metrics["MAE"]) - float(exp_metrics["MAE"])
    percentage = absolute / float(prod_metrics["MAE"]) * 100.0
    verdict = "PROMOTION CANDIDATE" if percentage > 0 else "DO NOT PROMOTE"

    scored = compare.copy(); scored["production_cost_prediction"] = production_cost; scored["experiment_cost_prediction"] = candidate_cost
    bootstrap = paired_project_mae_comparison(scored, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost_prediction", candidate_prediction="experiment_cost_prediction", bootstrap_samples=5000, seed=48000 + int(training_end))
    comparison = {
        "production_cost_mae": prod_metrics["MAE"], "experiment_cost_mae": exp_metrics["MAE"],
        "absolute_cost_mae_improvement": round(absolute, 6), "cost_improvement_percentage": round(percentage, 6),
        "production_delay_mae": delay_metrics["MAE"], "experiment_delay_mae": delay_metrics["MAE"], "delay_predictions_identical": True,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(compare)),
        "paired_project_bootstrap": bootstrap, "diagnostics": _diagnostics(scored),
        "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict,
    }
    run_id = f"exp48-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"; window = f"{training_start}_{training_end}"
    ledger = build_prediction_ledger(scored, experiment_id=EXPERIMENT_ID, window=window, production_cost_prediction=production_cost, experiment_cost_prediction=candidate_cost, extra_columns=["completion_year", "lifecycle_stage", "sector", "implementing_agency", "state", "project_size_category", "approved_cost_cr", "cost_escalation_percentage", "revised_cost_cr", "cumulative_expenditure_cr", "exp12_history_12m", "exp34_observations_seen"])
    assert_prediction_ledger_matches_cohort(ledger, compare)
    persisted = write_experiment_prediction_ledger(ledger, experiment_id=EXPERIMENT_ID, window=window, run_id=run_id, extra_manifest={"primary_target": "cost", "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict, "changed_dimension": CHANGED_DIMENSION, "bootstrap_samples": 5000, "delay_unchanged": True})
    artifacts = _write_artifacts(Path(persisted["ledger_path"]).parent, features=features, bootstrap=bootstrap, comparison=comparison)
    lookup = {(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): float(pred) for (_, row), pred in zip(scored.iterrows(), candidate_cost)}
    return {
        "experiment": {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE, "run_id": run_id, "model_role": "experiment", "promotion_allowed": False, "changed_dimension": CHANGED_DIMENSION, "hypothesis": HYPOTHESIS, "new_features": EXP48_FEATURES, "production_cost_family": family, "calibration_method": "production Exp33 rolling-OOF residual method refit for challenger", "rolling_oof": oof, "calibration": _public_calibration(calibration), "future_holdout_used_for_selection_or_calibration": False, "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict, "ledger_path": str(persisted["ledger_path"]), "ledger_manifest_path": str(persisted["manifest_path"]), "cohort_fingerprint": persisted["manifest"]["cohort_fingerprint"], "ledger_fingerprint": persisted["manifest"]["ledger_fingerprint"], "audit_artifacts": artifacts},
        "overall_comparison": comparison,
        "state": {"lookup": lookup, "features": features, "candidate_model": candidate_model, "calibration": calibration},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    keys = set(state.get("lookup", {})); mask = [(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) in keys for _, row in frame.iterrows()]
    return assign_project_balanced_weights(frame.loc[mask].copy())


def predict_project(row: pd.Series, state: dict) -> dict:
    key = (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
    if key not in state.get("lookup", {}): raise ValueError("Exp48 row is outside the frozen comparison cohort")
    return {"cost_overrun_percentage": float(state["lookup"][key])}
