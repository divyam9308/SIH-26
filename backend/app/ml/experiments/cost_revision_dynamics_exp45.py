"""Experiment 45: causal Cost revision dynamics and deterministic change points.

The production Cost target, ExtraTrees family, project weighting, temporal split,
and Exp33 rolling-OOF residual calibration method are retained.  The sole
scientific change is a fixed, interpretable family of all-history Cost revision
features built from the official report prefix available at each snapshot.
"""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _corrections,
    _cost_calibration_oof,
    _public_calibration,
)
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import (
    assert_prediction_ledger_matches_cohort,
    build_prediction_ledger,
    write_experiment_prediction_ledger,
)
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
)
from backend.app.ml.production_exp35_baseline import (
    CALIBRATION_GATE_FEATURE,
    _select_aft_calibration_projects,
)

EXPERIMENT_ID = "exp_45"
EXPERIMENT_SEQUENCE = 45
EXPERIMENT_NAME = "Cost revision dynamics and change-point features"
EXPERIMENT_SCOPE = "cost"
HYPOTHESIS = (
    "The frequency, direction, timing, burstiness and regime structure of the "
    "complete as-of Cost revision process adds signal beyond Exp12's short-window trajectories."
)

# This fixed family was specified before seeing either future holdout.  It does
# not duplicate Exp12's trailing revision count/magnitude/recency features.
EXP45_FEATURES = [
    "exp45_revision_count_total",
    "exp45_revision_count_positive",
    "exp45_revision_count_negative",
    "exp45_cumulative_positive_revision_pp",
    "exp45_cumulative_negative_revision_pp",
    "exp45_cumulative_absolute_revision_pp",
    "exp45_net_revision_pp",
    "exp45_last_revision_pp",
    "exp45_absolute_last_revision_pp",
    "exp45_months_since_last_revision",
    "exp45_max_absolute_revision_pp",
    "exp45_largest_revision_sign",
    "exp45_months_since_largest_revision",
    "exp45_mean_inter_revision_months",
    "exp45_inter_revision_cv",
    "exp45_max_revisions_any_6m",
    "exp45_recent_12m_revision_fraction",
    "exp45_revision_cluster_score",
    "exp45_same_direction_streak",
    "exp45_revision_reversal_flag",
    "exp45_positive_revision_share",
    "exp45_revision_direction_entropy",
    "exp45_change_point_score",
    "exp45_change_point_direction",
    "exp45_months_since_change_point",
    "exp45_change_point_slope_difference",
]

FORBIDDEN_INPUTS = {
    "completion_date",
    "actual_completion_date",
    "actual_cost_overrun_percentage",
    "final_cost",
    "final_revised_cost",
}
SOURCE_COLUMNS = {
    "canonical_project_id",
    "snapshot_date",
    "approved_cost_cr",
    "revised_cost_cr",
}


def _months(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return float((later - earlier).days / 30.4375)


def _slope(dates: list[pd.Timestamp], values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.array([(date - dates[0]).days / 30.4375 for date in dates], dtype=float)
    y = np.asarray(values, dtype=float)
    if np.ptp(x) <= 1e-12:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _change_point(prefix_dates: list[pd.Timestamp], prefix_levels: list[float]) -> tuple[float, float, float, float]:
    """Strongest robust level shift in the trailing 24-month prefix."""
    if len(prefix_levels) < 6:
        return 0.0, 0.0, -1.0, 0.0
    current = prefix_dates[-1]
    keep = [i for i, date in enumerate(prefix_dates) if (current - date).days <= 731]
    dates = [prefix_dates[i] for i in keep]
    levels = [prefix_levels[i] for i in keep]
    if len(levels) < 6:
        return 0.0, 0.0, -1.0, 0.0

    array = np.asarray(levels, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median))) * 1.4826
    scale = max(mad, 1.0)
    best = (0.0, 0.0, -1.0, 0.0)
    for split in range(3, len(levels) - 2):
        pre, post = array[:split], array[split:]
        shift = float(np.median(post) - np.median(pre))
        score = abs(shift) / scale
        if score > best[0] + 1e-12:
            pre_slope = _slope(dates[:split], pre.tolist())
            post_slope = _slope(dates[split:], post.tolist())
            best = (
                float(score),
                float(np.sign(shift)),
                _months(current, dates[split]),
                float(post_slope - pre_slope),
            )
    return best


def _canonical_history(history: pd.DataFrame) -> pd.DataFrame:
    required = sorted(SOURCE_COLUMNS)
    missing = [column for column in required if column not in history]
    if missing:
        raise ValueError("Exp45 history is missing: " + ", ".join(missing))
    frame = history.copy()
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame["approved_cost_cr"] = pd.to_numeric(frame["approved_cost_cr"], errors="coerce")
    frame["revised_cost_cr"] = pd.to_numeric(frame["revised_cost_cr"], errors="coerce")
    frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"])
    # Differing duplicate reports are resolved independently of input order by a
    # deterministic content hash.  One canonical row remains per month/date.
    # Tie-breaking must itself be prediction-time safe: hash only declared as-of
    # source fields, never labels or completion outcomes carried by the history.
    hash_frame = frame.reindex(columns=sorted(SOURCE_COLUMNS)).astype("string").fillna("<NA>")
    frame["_exp45_tie"] = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(np.uint64)
    frame = frame.sort_values(
        ["canonical_project_id", "snapshot_date", "_exp45_tie"], kind="mergesort"
    ).drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    return frame.drop(columns="_exp45_tie").reset_index(drop=True)


def engineer_revision_history(history: pd.DataFrame) -> pd.DataFrame:
    """Create prefix-only revision-process features for every official report."""
    frame = _canonical_history(history)
    records: list[dict[str, object]] = []

    for _, group in frame.groupby("canonical_project_id", sort=False):
        event_dates: list[pd.Timestamp] = []
        increments: list[float] = []
        level_dates: list[pd.Timestamp] = []
        levels: list[float] = []
        previous_revised = math.nan

        for _, row in group.iterrows():
            current_date = pd.Timestamp(row["snapshot_date"])
            approved = float(row["approved_cost_cr"]) if pd.notna(row["approved_cost_cr"]) else math.nan
            revised = float(row["revised_cost_cr"]) if pd.notna(row["revised_cost_cr"]) else math.nan
            if np.isfinite(approved) and approved > 0 and np.isfinite(revised):
                level_dates.append(current_date)
                levels.append((revised - approved) / approved * 100.0)
                if np.isfinite(previous_revised) and abs(revised - previous_revised) > 1e-9:
                    event_dates.append(current_date)
                    increments.append((revised - previous_revised) / approved * 100.0)
            if np.isfinite(revised):
                previous_revised = revised

            values = np.asarray(increments, dtype=float)
            signs = np.sign(values)
            positive = values[values > 0]
            negative = values[values < 0]
            count = len(values)
            record: dict[str, object] = {
                "canonical_project_id": row["canonical_project_id"],
                "snapshot_date": current_date,
                **{feature: 0.0 for feature in EXP45_FEATURES},
                "exp45_months_since_last_revision": -1.0,
                "exp45_months_since_largest_revision": -1.0,
                "exp45_months_since_change_point": -1.0,
                "exp45_revision_count_total": count,
                "exp45_revision_count_positive": len(positive),
                "exp45_revision_count_negative": len(negative),
                "exp45_cumulative_positive_revision_pp": positive.sum() if len(positive) else 0.0,
                "exp45_cumulative_negative_revision_pp": negative.sum() if len(negative) else 0.0,
                "exp45_cumulative_absolute_revision_pp": np.abs(values).sum() if count else 0.0,
                "exp45_net_revision_pp": values.sum() if count else 0.0,
            }

            if count:
                record["exp45_last_revision_pp"] = values[-1]
                record["exp45_absolute_last_revision_pp"] = abs(values[-1])
                record["exp45_months_since_last_revision"] = _months(current_date, event_dates[-1])
                largest = int(np.argmax(np.abs(values)))
                record["exp45_max_absolute_revision_pp"] = abs(values[largest])
                record["exp45_largest_revision_sign"] = signs[largest]
                record["exp45_months_since_largest_revision"] = _months(current_date, event_dates[largest])
                record["exp45_positive_revision_share"] = len(positive) / count
                p = len(positive) / count
                record["exp45_revision_direction_entropy"] = (
                    0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
                )
                streak = 1
                for pos in range(count - 2, -1, -1):
                    if signs[pos] != signs[-1]:
                        break
                    streak += 1
                record["exp45_same_direction_streak"] = streak
                record["exp45_revision_reversal_flag"] = float(count >= 2 and signs[-1] != signs[-2])

                recent12 = sum((current_date - date).days <= 366 for date in event_dates)
                record["exp45_recent_12m_revision_fraction"] = recent12 / count
                max6 = 0
                for end_date in event_dates:
                    max6 = max(max6, sum(0 <= (end_date - date).days <= 183 for date in event_dates))
                record["exp45_max_revisions_any_6m"] = max6
                record["exp45_revision_cluster_score"] = max6 / math.sqrt(count)

            if count >= 2:
                intervals = np.diff(np.array([date.value for date in event_dates], dtype=np.int64)) / 86_400_000_000_000 / 30.4375
                mean_interval = float(np.mean(intervals))
                record["exp45_mean_inter_revision_months"] = mean_interval
                record["exp45_inter_revision_cv"] = float(np.std(intervals) / mean_interval) if mean_interval > 1e-12 else 0.0

            cp_score, cp_direction, cp_months, cp_slope = _change_point(level_dates, levels)
            record["exp45_change_point_score"] = cp_score
            record["exp45_change_point_direction"] = cp_direction
            record["exp45_months_since_change_point"] = cp_months
            record["exp45_change_point_slope_difference"] = cp_slope
            records.append(record)

    return pd.DataFrame.from_records(records, columns=["canonical_project_id", "snapshot_date", *EXP45_FEATURES])


def enrich_revision_dynamics(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = pd.read_csv(TRAJECTORIES, low_memory=False) if TRAJECTORIES.exists() else supervised.copy()
    engineered = engineer_revision_history(history)
    rows = supervised.copy()
    rows["canonical_project_id"] = rows["canonical_project_id"].astype("string")
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    result = rows.merge(
        engineered,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    if len(result) != len(rows):
        raise AssertionError("Exp45 feature engineering changed the supervised cohort")
    for feature in EXP45_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(
            -1.0 if feature.startswith("exp45_months_since_") else 0.0
        )
    return result


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


def _diagnostics(rows: pd.DataFrame, actual: str, baseline: str, candidate: str) -> dict:
    result: dict[str, object] = {}
    work = rows.copy()
    work["_baseline_error"] = (work[baseline] - work[actual]).abs()
    work["_candidate_error"] = (work[candidate] - work[actual]).abs()
    per_project = work.groupby("canonical_project_id", sort=True).agg(
        baseline_mae=("_baseline_error", "mean"), candidate_mae=("_candidate_error", "mean")
    )
    result["median_per_project_mae"] = {
        "production": float(per_project.baseline_mae.median()),
        "experiment": float(per_project.candidate_mae.median()),
    }
    result["p90_per_project_mae"] = {
        "production": float(per_project.baseline_mae.quantile(0.9)),
        "experiment": float(per_project.candidate_mae.quantile(0.9)),
    }
    result["absolute_error_p90"] = {
        "production": float(work._baseline_error.quantile(0.9)),
        "experiment": float(work._candidate_error.quantile(0.9)),
    }
    stages = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = work[work.get("lifecycle_stage", pd.Series(index=work.index, dtype="string")).eq(stage)]
        if part.empty:
            stages[stage] = {"snapshots": 0, "projects": 0, "production_mae": None, "experiment_mae": None}
            continue
        weights = part["sample_weight"].to_numpy(float)
        stages[stage] = {
            "snapshots": int(len(part)),
            "projects": int(part.canonical_project_id.nunique()),
            "production_mae": float(np.average(part._baseline_error, weights=weights)),
            "experiment_mae": float(np.average(part._candidate_error, weights=weights)),
        }
    result["lifecycle_mae"] = stages
    return result


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_revision_dynamics(
        enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    )
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    overlap = set(train.canonical_project_id) & set(test.canonical_project_id)
    if overlap:
        raise ValueError(f"Exp45 train/holdout project overlap: {len(overlap)}")

    production_cost_model = production_bundle["cost"]
    base_features = list(production_cost_model.features)
    duplicated = sorted(set(base_features) & set(EXP45_FEATURES))
    if duplicated:
        raise ValueError("Exp45 duplicates production features: " + ", ".join(duplicated))
    features = list(dict.fromkeys(base_features + EXP45_FEATURES))
    family = _cost_family(production_cost_model)

    calibration, oof = _cost_calibration_oof(train, features, family)
    candidate_model = _fit_pipeline(
        _regressors(PRODUCTION_COST_SEED)[family], train, features, "actual_cost_overrun_percentage"
    )

    compare = _production_cost_evaluation_rows(test)
    # Recreate the evidence-only historical gate used by the freshly trained
    # production Delay wrapper so unchanged-target verification is exact.
    gate_projects = _select_aft_calibration_projects(compare)
    compare = compare.copy()
    compare[CALIBRATION_GATE_FEATURE] = compare.canonical_project_id.astype("string").isin(gate_projects)
    compare = assign_project_balanced_weights(compare)

    production_cost = production_cost_model.predict(compare)
    raw_candidate_cost = candidate_model.predict(compare[features])
    candidate_cost = raw_candidate_cost + _corrections(compare, raw_candidate_cost, calibration)
    production_delay = np.maximum(0.0, production_bundle["delay"].predict(compare))
    candidate_delay = production_delay.copy()
    if not np.array_equal(production_delay, candidate_delay):
        raise AssertionError("Exp45 changed Delay predictions")

    prod_metrics = _regression_metrics(
        compare.actual_cost_overrun_percentage, production_cost, compare.sample_weight, compare.canonical_project_id
    )
    exp_metrics = _regression_metrics(
        compare.actual_cost_overrun_percentage, candidate_cost, compare.sample_weight, compare.canonical_project_id
    )
    delay_metrics = _regression_metrics(
        compare.actual_delay_days, production_delay, compare.sample_weight, compare.canonical_project_id
    )
    absolute_improvement = float(prod_metrics["MAE"]) - float(exp_metrics["MAE"])
    improvement_pct = absolute_improvement / float(prod_metrics["MAE"]) * 100.0
    scientific_verdict = "PROMOTION CANDIDATE" if improvement_pct > 0 else "DO NOT PROMOTE"

    scored = compare.copy()
    scored["production_cost_prediction"] = production_cost
    scored["experiment_cost_prediction"] = candidate_cost
    statistics = paired_project_mae_comparison(
        scored,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost_prediction",
        candidate_prediction="experiment_cost_prediction",
        bootstrap_samples=5000,
        seed=45000 + int(training_end),
    )
    diagnostics = _diagnostics(
        scored, "actual_cost_overrun_percentage", "production_cost_prediction", "experiment_cost_prediction"
    )

    run_id = f"exp45-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    window = f"{training_start}_{training_end}"
    ledger = build_prediction_ledger(
        scored,
        experiment_id=EXPERIMENT_ID,
        window=window,
        production_cost_prediction=production_cost,
        experiment_cost_prediction=candidate_cost,
        extra_columns=[
            "completion_year", "lifecycle_stage", "sector", "implementing_agency", "state",
            "project_size_category", "approved_cost_cr", "cost_escalation_percentage",
            "revised_cost_cr", "cumulative_expenditure_cr", "exp12_history_12m",
            "exp34_observations_seen", "parser_family",
        ],
    )
    assert_prediction_ledger_matches_cohort(ledger, compare)
    persisted = write_experiment_prediction_ledger(
        ledger,
        experiment_id=EXPERIMENT_ID,
        window=window,
        run_id=run_id,
        extra_manifest={
            "primary_target": "cost",
            "execution_verdict": "EXECUTION VALID",
            "scientific_verdict": scientific_verdict,
            "changed_dimension": "feature_set",
            "bootstrap_samples": 5000,
            "delay_unchanged": True,
        },
    )

    lookup = {
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): float(prediction)
        for (_, row), prediction in zip(scored.iterrows(), candidate_cost)
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": run_id,
            "model_role": "experiment",
            "promotion_allowed": False,
            "changed_dimension": "feature_set",
            "hypothesis": HYPOTHESIS,
            "feature_count": len(EXP45_FEATURES),
            "new_features": EXP45_FEATURES,
            "production_cost_family": family,
            "calibration_method": "production Exp33 rolling-OOF weighted-median residual calibration, refit for challenger",
            "rolling_oof": oof,
            "calibration": _public_calibration(calibration),
            "future_holdout_used_for_selection_or_calibration": False,
            "execution_verdict": "EXECUTION VALID",
            "scientific_verdict": scientific_verdict,
            "ledger_path": str(persisted["ledger_path"]),
            "ledger_manifest_path": str(persisted["manifest_path"]),
            "cohort_fingerprint": persisted["manifest"]["cohort_fingerprint"],
            "ledger_fingerprint": persisted["manifest"]["ledger_fingerprint"],
        },
        "overall_comparison": {
            "production_cost_mae": prod_metrics["MAE"],
            "experiment_cost_mae": exp_metrics["MAE"],
            "absolute_cost_mae_improvement": round(absolute_improvement, 6),
            "cost_improvement_percentage": round(improvement_pct, 6),
            "production_delay_mae": delay_metrics["MAE"],
            "experiment_delay_mae": delay_metrics["MAE"],
            "delay_predictions_identical": True,
            "comparison_test_projects": int(compare.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "paired_project_bootstrap": statistics,
            "diagnostics": diagnostics,
            "execution_verdict": "EXECUTION VALID",
            "scientific_verdict": scientific_verdict,
        },
        "state": {"lookup": lookup, "features": features, "candidate_model": candidate_model, "calibration": calibration},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    keys = set(state.get("lookup", {}))
    mask = [
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) in keys
        for _, row in frame.iterrows()
    ]
    return assign_project_balanced_weights(frame.loc[mask].copy())


def predict_project(row: pd.Series, state: dict) -> dict:
    key = (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
    if key not in state.get("lookup", {}):
        raise ValueError("Exp45 row is outside the frozen comparison cohort")
    return {"cost_overrun_percentage": float(state["lookup"][key])}
