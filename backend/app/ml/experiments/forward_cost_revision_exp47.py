"""Experiment 47: cross-fitted forward Cost-revision representation.

The final target, production Cost family, temporal split, project weighting and
Exp33 calibration method are unchanged.  The only scientific change is a set of
strictly cross-fitted predictions for recurrent *intermediate* Cost revisions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import _corrections, _cost_calibration_oof, _public_calibration
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import (
    assert_prediction_ledger_matches_cohort,
    build_prediction_ledger,
    write_experiment_prediction_ledger,
)
from backend.app.ml.experiments.trajectory_exp12 import EXP12_FEATURES, engineer_history
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _json_safe, _preprocessor, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED, _production_cost_evaluation_rows, enrich_supervised_for_production
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE, _select_aft_calibration_projects

EXPERIMENT_ID = "exp_47"
EXPERIMENT_SEQUENCE = 47
EXPERIMENT_NAME = "Full-archive forward Cost-revision representation"
EXPERIMENT_SCOPE = "cost"
HYPOTHESIS = (
    "Intermediate future Cost-revision events learned from the full safe monthly archive "
    "provide information beyond backward-looking Exp12 trajectory summaries."
)
CHANGED_DIMENSION = "cross_fitted_intermediate_supervision"
MIN_REVISION_PP = 0.25
AUXILIARY_FOLDS = 3
AUXILIARY_SEED = 47047
HORIZON_DAYS = {3: 92, 6: 183, 12: 366}

FORBIDDEN_AUX_INPUTS = {
    "completion_date", "actual_completion_date", "actual_cost_overrun_percentage",
    "reported_completion_expenditure_cr", "actual_delay_days", "actual_risk",
}
AS_OF_SOURCE_COLUMNS = [
    "canonical_project_id", "snapshot_date", "approved_cost_cr", "revised_cost_cr",
    "cumulative_expenditure_cr", "schedule_slippage_days", "duration_ratio", "sector",
    "project_size_category", "current_schedule_status", "implementing_agency",
]
AUXILIARY_INPUT_FEATURES = [
    "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr",
    "cost_escalation_percentage", "expenditure_ratio", "schedule_slippage_days",
    "duration_ratio", "sector", "project_size_category", "current_schedule_status",
    "exp47_history_observations", "exp47_days_since_previous_report",
    "exp47_prior_revision_count", "exp47_months_since_prior_revision",
    "exp47_previous_revision_pp", "exp47_expenditure_ratio_change",
    *EXP12_FEATURES,
]
EXP47_FEATURES = [
    "exp47_revision_probability_3m", "exp47_revision_probability_6m",
    "exp47_revision_probability_12m", "exp47_days_to_next_revision_prediction",
    "exp47_next_revision_magnitude_pp_prediction", "exp47_positive_revision_probability",
]
AUXILIARY_LABELS = [
    "cost_revision_within_3m", "cost_revision_within_6m", "cost_revision_within_12m",
    "days_to_next_cost_revision", "next_cost_revision_pp", "next_cost_revision_positive",
]


class _ConstantClassifier:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, frame):
        p = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class _ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def _canonical_archive(history: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"canonical_project_id", "snapshot_date", "approved_cost_cr", "revised_cost_cr"}.difference(history.columns))
    if missing:
        raise ValueError("Exp47 archive is missing: " + ", ".join(missing))
    frame = history.copy()
    for column in AS_OF_SOURCE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string").str.strip()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    for column in ("approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "schedule_slippage_days", "duration_ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"])
    frame = frame[frame["canonical_project_id"].ne("")].copy()
    # The tie breaker hashes as-of source fields only. Outcome columns carried by
    # the resolved archive can never influence which duplicate report survives.
    safe = frame.reindex(columns=AS_OF_SOURCE_COLUMNS).astype("string").fillna("<NA>")
    frame["_exp47_tie"] = pd.util.hash_pandas_object(safe, index=False).to_numpy(np.uint64)
    frame = frame.sort_values(["canonical_project_id", "snapshot_date", "_exp47_tie"], kind="mergesort")
    return frame.drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last").drop(columns="_exp47_tie").reset_index(drop=True)


def _add_prefix_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = engineer_history(frame)
    for feature in (
        "exp47_history_observations", "exp47_days_since_previous_report",
        "exp47_prior_revision_count", "exp47_months_since_prior_revision",
        "exp47_previous_revision_pp", "exp47_expenditure_ratio_change",
    ):
        result[feature] = 0.0
    result["exp47_days_since_previous_report"] = -1.0
    result["exp47_months_since_prior_revision"] = -1.0
    for _, group in result.groupby("canonical_project_id", sort=False):
        indices = group.index.to_numpy()
        dates = group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        approved = pd.to_numeric(group["approved_cost_cr"], errors="coerce").to_numpy(float)
        revised = pd.to_numeric(group["revised_cost_cr"], errors="coerce").to_numpy(float)
        spend = pd.to_numeric(group["cumulative_expenditure_cr"], errors="coerce").to_numpy(float)
        prior_count = 0
        last_event = -1
        last_pp = 0.0
        previous_ratio = math.nan
        for position, index in enumerate(indices):
            result.at[index, "exp47_history_observations"] = position + 1
            if position:
                result.at[index, "exp47_days_since_previous_report"] = float((dates[position] - dates[position - 1]) / np.timedelta64(1, "D"))
                if np.isfinite(revised[position]) and np.isfinite(revised[position - 1]) and np.isfinite(approved[position]) and approved[position] > 0:
                    pp = 100.0 * (revised[position] - revised[position - 1]) / approved[position]
                    if abs(pp) >= MIN_REVISION_PP:
                        prior_count += 1
                        last_event = position
                        last_pp = pp
            ratio = spend[position] / approved[position] if np.isfinite(spend[position]) and np.isfinite(approved[position]) and approved[position] > 0 else math.nan
            result.at[index, "exp47_prior_revision_count"] = prior_count
            result.at[index, "exp47_previous_revision_pp"] = last_pp
            if last_event >= 0:
                result.at[index, "exp47_months_since_prior_revision"] = float((dates[position] - dates[last_event]) / np.timedelta64(1, "D") / 30.4375)
            if np.isfinite(ratio) and np.isfinite(previous_ratio):
                result.at[index, "exp47_expenditure_ratio_change"] = ratio - previous_ratio
            if np.isfinite(ratio):
                previous_ratio = ratio
    if "cost_escalation_percentage" not in result:
        result["cost_escalation_percentage"] = np.where(
            result["approved_cost_cr"].gt(0),
            (result["revised_cost_cr"] - result["approved_cost_cr"]) / result["approved_cost_cr"] * 100.0,
            np.nan,
        )
    if "expenditure_ratio" not in result:
        result["expenditure_ratio"] = np.where(
            result["approved_cost_cr"].gt(0), result["cumulative_expenditure_cr"] / result["approved_cost_cr"], np.nan
        )
    return result


def build_forward_cost_revision_dataset(history: pd.DataFrame, *, cutoff: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Build as-of inputs and forward labels; truncated histories remain censored."""
    frame = _canonical_archive(history)
    if cutoff is not None:
        frame = frame[frame["snapshot_date"].le(pd.Timestamp(cutoff))].copy()
    frame = _add_prefix_features(frame)
    for label in AUXILIARY_LABELS:
        frame[label] = np.nan
    frame["auxiliary_followup_days"] = 0.0
    frame["auxiliary_next_revision_observed"] = 0.0

    for _, group in frame.groupby("canonical_project_id", sort=False):
        indices = group.index.to_numpy()
        dates = group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        approved = pd.to_numeric(group["approved_cost_cr"], errors="coerce").to_numpy(float)
        revised = pd.to_numeric(group["revised_cost_cr"], errors="coerce").to_numpy(float)
        event_positions: list[int] = []
        event_pp: list[float] = []
        for position in range(1, len(group)):
            if np.isfinite(revised[position]) and np.isfinite(revised[position - 1]) and np.isfinite(approved[position]) and approved[position] > 0:
                pp = 100.0 * (revised[position] - revised[position - 1]) / approved[position]
                if abs(pp) >= MIN_REVISION_PP:
                    event_positions.append(position)
                    event_pp.append(float(pp))
        event_array = np.asarray(event_positions, dtype=int)
        for position, index in enumerate(indices):
            followup = float((dates[-1] - dates[position]) / np.timedelta64(1, "D"))
            frame.at[index, "auxiliary_followup_days"] = max(0.0, followup)
            pointer = int(np.searchsorted(event_array, position + 1)) if len(event_array) else 0
            has_event = pointer < len(event_positions)
            days = float((dates[event_positions[pointer]] - dates[position]) / np.timedelta64(1, "D")) if has_event else math.nan
            for months, horizon in HORIZON_DAYS.items():
                label = f"cost_revision_within_{months}m"
                if has_event and days <= horizon:
                    frame.at[index, label] = 1.0
                elif followup >= horizon:
                    frame.at[index, label] = 0.0
            if has_event:
                pp = event_pp[pointer]
                frame.at[index, "auxiliary_next_revision_observed"] = 1.0
                frame.at[index, "days_to_next_cost_revision"] = days
                frame.at[index, "next_cost_revision_pp"] = pp
                frame.at[index, "next_cost_revision_positive"] = float(pp > 0)
    return frame


def _fit_classifier(rows: pd.DataFrame, label: str, seed: int):
    available = rows.dropna(subset=[label]).copy()
    if available.empty:
        return _ConstantClassifier(0.0)
    available = assign_project_balanced_weights(available)
    probability = float(available[label].mean())
    if available[label].nunique() < 2:
        return _ConstantClassifier(probability)
    model = ExtraTreesClassifier(
        n_estimators=120, min_samples_leaf=5, max_features=0.8,
        class_weight="balanced_subsample", random_state=seed, n_jobs=2,
    )
    pipe = Pipeline([("preprocess", _preprocessor(available, AUXILIARY_INPUT_FEATURES)), ("model", model)])
    pipe.fit(available[AUXILIARY_INPUT_FEATURES], available[label].astype(int), model__sample_weight=available.sample_weight.to_numpy(float))
    return pipe


def _fit_regressor(rows: pd.DataFrame, label: str, seed: int):
    available = rows.dropna(subset=[label]).copy()
    if available.empty:
        return _ConstantRegressor(0.0)
    available = assign_project_balanced_weights(available)
    if available[label].nunique() < 2:
        return _ConstantRegressor(float(available[label].median()))
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5, max_features=0.8, random_state=seed, n_jobs=2)
    pipe = Pipeline([("preprocess", _preprocessor(available, AUXILIARY_INPUT_FEATURES)), ("model", model)])
    pipe.fit(available[AUXILIARY_INPUT_FEATURES], available[label], model__sample_weight=available.sample_weight.to_numpy(float))
    return pipe


def fit_auxiliary_models(rows: pd.DataFrame) -> dict[str, object]:
    return {
        "event_3m": _fit_classifier(rows, "cost_revision_within_3m", AUXILIARY_SEED + 3),
        "event_6m": _fit_classifier(rows, "cost_revision_within_6m", AUXILIARY_SEED + 6),
        "event_12m": _fit_classifier(rows, "cost_revision_within_12m", AUXILIARY_SEED + 12),
        "days": _fit_regressor(rows, "days_to_next_cost_revision", AUXILIARY_SEED + 20),
        "magnitude": _fit_regressor(rows, "next_cost_revision_pp", AUXILIARY_SEED + 21),
        "positive": _fit_classifier(rows, "next_cost_revision_positive", AUXILIARY_SEED + 22),
    }


def _positive_probability(model, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(rows[AUXILIARY_INPUT_FEATURES]))[:, 1]


def predict_auxiliary(models: dict[str, object], rows: pd.DataFrame) -> pd.DataFrame:
    result = rows[["canonical_project_id", "snapshot_date"]].copy()
    result["exp47_revision_probability_3m"] = _positive_probability(models["event_3m"], rows)
    result["exp47_revision_probability_6m"] = _positive_probability(models["event_6m"], rows)
    result["exp47_revision_probability_12m"] = _positive_probability(models["event_12m"], rows)
    result["exp47_days_to_next_revision_prediction"] = np.maximum(0.0, models["days"].predict(rows[AUXILIARY_INPUT_FEATURES]))
    result["exp47_next_revision_magnitude_pp_prediction"] = models["magnitude"].predict(rows[AUXILIARY_INPUT_FEATURES])
    result["exp47_positive_revision_probability"] = _positive_probability(models["positive"], rows)
    return result


def _rows_for_keys(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    keys = target[["canonical_project_id", "snapshot_date"]].copy()
    keys["canonical_project_id"] = keys["canonical_project_id"].astype("string")
    keys["snapshot_date"] = pd.to_datetime(keys["snapshot_date"], errors="coerce")
    columns = ["canonical_project_id", "snapshot_date", *AUXILIARY_INPUT_FEATURES, *AUXILIARY_LABELS]
    available = source.reindex(columns=columns).drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    merged = keys.merge(available, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")
    if len(merged) != len(target):
        raise AssertionError("Exp47 auxiliary row lookup changed the supervised cohort")
    return merged


def _auxiliary_diagnostics(rows: pd.DataFrame, predictions: pd.DataFrame, training_pool: pd.DataFrame) -> dict:
    result: dict[str, object] = {
        "meaningful_revision_threshold_pp": MIN_REVISION_PP,
        "training_rows": int(len(training_pool)),
        "training_projects": int(training_pool.canonical_project_id.nunique()),
        "observed_revision_events": int(
            training_pool.groupby("canonical_project_id")["exp47_prior_revision_count"].max().sum()
        ),
        "censored_or_no_next_event_rows": int((training_pool["auxiliary_next_revision_observed"] == 0).sum()),
    }
    mapping = {3: "exp47_revision_probability_3m", 6: "exp47_revision_probability_6m", 12: "exp47_revision_probability_12m"}
    for months, prediction in mapping.items():
        label = f"cost_revision_within_{months}m"
        mask = rows[label].notna()
        y = rows.loc[mask, label].to_numpy(float)
        p = predictions.loc[mask, prediction].to_numpy(float)
        item = {"available_rows": int(mask.sum()), "prevalence": float(y.mean()) if len(y) else None}
        if len(np.unique(y)) > 1:
            item.update({"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "brier_score": float(brier_score_loss(y, p))})
        result[f"event_{months}m"] = item
    for label, prediction, name in (
        ("days_to_next_cost_revision", "exp47_days_to_next_revision_prediction", "time_to_next_revision"),
        ("next_cost_revision_pp", "exp47_next_revision_magnitude_pp_prediction", "next_revision_magnitude"),
    ):
        mask = rows[label].notna()
        result[name] = {"available_rows": int(mask.sum()), "mae": float(mean_absolute_error(rows.loc[mask, label], predictions.loc[mask, prediction])) if mask.any() else None}
    return _json_safe(result)


def cross_fitted_auxiliary_features(
    history: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    training_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cutoff = pd.Timestamp(year=int(training_end), month=12, day=31)
    training_archive = build_forward_cost_revision_dataset(history, cutoff=cutoff)
    prediction_archive = build_forward_cost_revision_dataset(history)
    holdout_ids = set(holdout["canonical_project_id"].astype("string"))
    pool = training_archive[~training_archive["canonical_project_id"].astype("string").isin(holdout_ids)].copy()
    if set(pool.canonical_project_id.astype("string")) & holdout_ids:
        raise AssertionError("Exp47 holdout project entered auxiliary training")

    train_rows = _rows_for_keys(training_archive, train)
    holdout_rows = _rows_for_keys(prediction_archive, holdout)
    projects = np.asarray(sorted(train["canonical_project_id"].astype("string").unique()))
    rng = np.random.default_rng(AUXILIARY_SEED + int(training_end))
    rng.shuffle(projects)
    folds = [part for part in np.array_split(projects, min(AUXILIARY_FOLDS, len(projects))) if len(part)]
    oof_parts: list[pd.DataFrame] = []
    for fold_projects in folds:
        fold_ids = set(fold_projects.tolist())
        fitting = pool[~pool["canonical_project_id"].astype("string").isin(fold_ids)].copy()
        validation = train_rows[train_rows["canonical_project_id"].astype("string").isin(fold_ids)].copy()
        models = fit_auxiliary_models(fitting)
        predicted = predict_auxiliary(models, validation)
        predicted["_exp47_row_order"] = validation.index.to_numpy()
        oof_parts.append(predicted)
    train_features = pd.concat(oof_parts, ignore_index=True).sort_values("_exp47_row_order").drop(columns="_exp47_row_order").reset_index(drop=True)
    if len(train_features) != len(train) or train_features.duplicated(["canonical_project_id", "snapshot_date"]).any():
        raise AssertionError("Exp47 OOF auxiliary predictions do not match final training rows")
    final_models = fit_auxiliary_models(pool)
    holdout_features = predict_auxiliary(final_models, holdout_rows).reset_index(drop=True)
    diagnostics = _auxiliary_diagnostics(train_rows.reset_index(drop=True), train_features, pool)
    diagnostics.update({
        "project_grouped_oof_folds": len(folds),
        "holdout_projects_excluded_from_auxiliary_training": len(holdout_ids),
        "auxiliary_training_cutoff": cutoff.date().isoformat(),
        "identity_rule": "stable canonical monthly history only; no fuzzy linking to final outcomes",
    })
    return train_features, holdout_features, _json_safe(diagnostics)


def _cost_family(production_model) -> str:
    inner = production_model.model.named_steps["model"]
    name = inner.__class__.__name__.lower()
    if "extratrees" in name:
        return "extra_trees"
    if "lgbm" in name:
        return "lightgbm"
    if "xgb" in name:
        return "xgboost"
    raise ValueError(f"Unsupported production Cost family: {inner.__class__.__name__}")


def _diagnostics(rows: pd.DataFrame) -> dict:
    work = rows.copy()
    work["_production_error"] = (work.production_cost_prediction - work.actual_cost_overrun_percentage).abs()
    work["_experiment_error"] = (work.experiment_cost_prediction - work.actual_cost_overrun_percentage).abs()
    per_project = work.groupby("canonical_project_id").agg(production=("_production_error", "mean"), experiment=("_experiment_error", "mean"))
    stages = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = work[work.get("lifecycle_stage", pd.Series(index=work.index, dtype="string")).eq(stage)]
        stages[stage] = {
            "projects": int(part.canonical_project_id.nunique()), "snapshots": int(len(part)),
            "production_mae": float(np.average(part._production_error, weights=part.sample_weight)) if len(part) else None,
            "experiment_mae": float(np.average(part._experiment_error, weights=part.sample_weight)) if len(part) else None,
        }
    return _json_safe({
        "median_per_project_mae": {"production": per_project.production.median(), "experiment": per_project.experiment.median()},
        "p90_per_project_mae": {"production": per_project.production.quantile(.9), "experiment": per_project.experiment.quantile(.9)},
        "absolute_error_p90": {"production": work._production_error.quantile(.9), "experiment": work._experiment_error.quantile(.9)},
        "lifecycle_mae": stages,
    })


def _write_audit_artifacts(directory: Path, *, auxiliary: dict, bootstrap: dict, features: list[str], comparison: dict) -> dict:
    payloads = {
        "feature_target_audit.json": {
            "changed_dimension": CHANGED_DIMENSION, "minimum_revision_pp": MIN_REVISION_PP,
            "auxiliary_inputs": AUXILIARY_INPUT_FEATURES, "auxiliary_labels": AUXILIARY_LABELS,
            "final_auxiliary_features": EXP47_FEATURES, "forbidden_inputs": sorted(FORBIDDEN_AUX_INPUTS),
            "future_reports_are_labels_only": True,
        },
        "auxiliary_model_diagnostics.json": auxiliary,
        "model_feature_config.json": {"auxiliary_family": "ExtraTrees fixed contract", "auxiliary_folds": AUXILIARY_FOLDS, "final_features": features},
        "bootstrap_results.json": bootstrap,
    }
    for filename, payload in payloads.items():
        (directory / filename).write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    summary = [
        "# Experiment 47 comparison", "", HYPOTHESIS, "",
        f"Production Cost MAE: {comparison['production_cost_mae']}",
        f"Experiment Cost MAE: {comparison['experiment_cost_mae']}",
        f"Improvement: {comparison['cost_improvement_percentage']}%", "",
        "Execution verdict: EXECUTION VALID", f"Scientific verdict: {comparison['scientific_verdict']}",
    ]
    (directory / "experiment_summary.md").write_text("\n".join(summary) + "\n")
    return {name: str(directory / name) for name in [*payloads, "experiment_summary.md"]}


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None, **_):
    if history is None:
        if not TRAJECTORIES.exists():
            raise FileNotFoundError("Exp47 requires the full monthly trajectory archive")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    base = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    base["completion_year"] = pd.to_numeric(base["completion_year"], errors="coerce")
    base["snapshot_date"] = pd.to_datetime(base["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(base, training_start, training_end, test_end)
    if set(train.canonical_project_id) & set(test.canonical_project_id):
        raise ValueError("Exp47 train/holdout project overlap")

    train_aux, test_aux, auxiliary_diagnostics = cross_fitted_auxiliary_features(history, train, test, training_end=training_end)
    train = train.merge(train_aux, on=["canonical_project_id", "snapshot_date"], how="left", validate="one_to_one")
    test = test.merge(test_aux, on=["canonical_project_id", "snapshot_date"], how="left", validate="one_to_one")
    for feature in EXP47_FEATURES:
        train[feature] = pd.to_numeric(train[feature], errors="coerce").fillna(0.0)
        test[feature] = pd.to_numeric(test[feature], errors="coerce").fillna(0.0)

    production_cost_model = production_bundle["cost"]
    base_features = list(production_cost_model.features)
    if set(base_features) & set(EXP47_FEATURES):
        raise ValueError("Exp47 auxiliary features already exist in production")
    features = list(dict.fromkeys(base_features + EXP47_FEATURES))
    family = _cost_family(production_cost_model)
    calibration, oof = _cost_calibration_oof(train, features, family)
    candidate_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family], train, features, "actual_cost_overrun_percentage")

    compare = _production_cost_evaluation_rows(test)
    gate_projects = _select_aft_calibration_projects(compare)
    compare = compare.copy()
    compare[CALIBRATION_GATE_FEATURE] = compare.canonical_project_id.astype("string").isin(gate_projects)
    compare = assign_project_balanced_weights(compare)
    production_cost = production_cost_model.predict(compare)
    raw_candidate = candidate_model.predict(compare[features])
    candidate_cost = raw_candidate + _corrections(compare, raw_candidate, calibration)
    production_delay = np.maximum(0.0, production_bundle["delay"].predict(compare))
    candidate_delay = production_delay.copy()
    if not np.array_equal(production_delay, candidate_delay):
        raise AssertionError("Exp47 changed Delay predictions")

    prod_metrics = _regression_metrics(compare.actual_cost_overrun_percentage, production_cost, compare.sample_weight, compare.canonical_project_id)
    exp_metrics = _regression_metrics(compare.actual_cost_overrun_percentage, candidate_cost, compare.sample_weight, compare.canonical_project_id)
    delay_metrics = _regression_metrics(compare.actual_delay_days, production_delay, compare.sample_weight, compare.canonical_project_id)
    absolute = float(prod_metrics["MAE"]) - float(exp_metrics["MAE"])
    percentage = absolute / float(prod_metrics["MAE"]) * 100.0

    scored = compare.copy()
    scored["production_cost_prediction"] = production_cost
    scored["experiment_cost_prediction"] = candidate_cost
    bootstrap = paired_project_mae_comparison(
        scored, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost_prediction",
        candidate_prediction="experiment_cost_prediction", bootstrap_samples=5000, seed=47000 + int(training_end),
    )
    supportive = bootstrap["probability_candidate_better"] >= 0.5
    verdict = "PROMOTION CANDIDATE" if percentage > 0 and supportive else "DO NOT PROMOTE"
    comparison = {
        "production_cost_mae": prod_metrics["MAE"], "experiment_cost_mae": exp_metrics["MAE"],
        "absolute_cost_mae_improvement": round(absolute, 6), "cost_improvement_percentage": round(percentage, 6),
        "production_delay_mae": delay_metrics["MAE"], "experiment_delay_mae": delay_metrics["MAE"],
        "delay_predictions_identical": True, "comparison_test_projects": int(compare.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(compare)), "paired_project_bootstrap": bootstrap,
        "diagnostics": _diagnostics(scored), "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict,
    }
    run_id = f"exp47-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    window = f"{training_start}_{training_end}"
    ledger = build_prediction_ledger(
        scored, experiment_id=EXPERIMENT_ID, window=window,
        production_cost_prediction=production_cost, experiment_cost_prediction=candidate_cost,
        extra_columns=[
            "completion_year", "lifecycle_stage", "sector", "implementing_agency", "state",
            "project_size_category", "approved_cost_cr", "cost_escalation_percentage", "revised_cost_cr",
            "cumulative_expenditure_cr", "exp12_history_12m", "exp34_observations_seen",
        ],
    )
    assert_prediction_ledger_matches_cohort(ledger, compare)
    persisted = write_experiment_prediction_ledger(
        ledger, experiment_id=EXPERIMENT_ID, window=window, run_id=run_id,
        extra_manifest={
            "primary_target": "cost", "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict,
            "changed_dimension": CHANGED_DIMENSION, "bootstrap_samples": 5000, "delay_unchanged": True,
            "auxiliary_training_projects": auxiliary_diagnostics["training_projects"],
            "auxiliary_observed_revision_events": auxiliary_diagnostics["observed_revision_events"],
        },
    )
    artifact_paths = _write_audit_artifacts(
        Path(persisted["ledger_path"]).parent, auxiliary=auxiliary_diagnostics,
        bootstrap=bootstrap, features=features, comparison=comparison,
    )
    lookup = {(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): float(pred) for (_, row), pred in zip(scored.iterrows(), candidate_cost)}
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": CHANGED_DIMENSION, "hypothesis": HYPOTHESIS, "new_features": EXP47_FEATURES,
            "production_cost_family": family, "auxiliary_model_family": "fixed ExtraTrees classifiers/regressors",
            "calibration_method": "production Exp33 rolling-OOF residual method refit on challenger OOF predictions",
            "rolling_oof": oof, "calibration": _public_calibration(calibration),
            "auxiliary_diagnostics": auxiliary_diagnostics, "future_holdout_used_for_selection_or_calibration": False,
            "execution_verdict": "EXECUTION VALID", "scientific_verdict": verdict,
            "ledger_path": str(persisted["ledger_path"]), "ledger_manifest_path": str(persisted["manifest_path"]),
            "cohort_fingerprint": persisted["manifest"]["cohort_fingerprint"], "ledger_fingerprint": persisted["manifest"]["ledger_fingerprint"],
            "audit_artifacts": artifact_paths,
        },
        "overall_comparison": comparison,
        "state": {"lookup": lookup, "features": features, "candidate_model": candidate_model, "calibration": calibration},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    keys = set(state.get("lookup", {}))
    mask = [(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) in keys for _, row in frame.iterrows()]
    return assign_project_balanced_weights(frame.loc[mask].copy())


def predict_project(row: pd.Series, state: dict) -> dict:
    key = (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
    if key not in state.get("lookup", {}):
        raise ValueError("Exp47 row is outside the frozen comparison cohort")
    return {"cost_overrun_percentage": float(state["lookup"][key])}
