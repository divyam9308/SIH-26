"""Experiment 13: advanced trajectory/regime interaction forecasting.

This challenger extends the promoted Experiment 12 trajectory representation with
past-only project-state context.  It keeps the production algorithms, temporal
split and weighting policy fixed, then asks an internal historical validation
block whether regime scores, cross-signal interactions, turning-point signals and
regime-transition features improve cost and/or delay forecasting.

The future holdout is never used for feature-group selection.  Production remains
unchanged; Experiment 13 writes only namespaced experiment artifacts.
"""
from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import (
    build_experiment_context,
    experiment_run_directory,
    new_experiment_manifest,
)
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.experiments.trajectory_exp12 import (
    MIN_HISTORY,
    WINDOW_DAYS,
    _algorithm,
    _key,
    _macro,
    _metric,
    _safe,
    _stage,
    _velocity,
)
from backend.app.ml.experiments.trajectory_exp12_v2 import engineer_history as engineer_exp12_history
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import target_feature_contract

EXPERIMENT_ID = "exp_13"
EXPERIMENT_NAME = "Advanced trajectory/regime interaction model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 13
IMPLEMENTATION_REVISION = "v1_regime_interactions"

REGIME_FEATURES = [
    "exp13_cost_pressure_score",
    "exp13_schedule_pressure_score",
    "exp13_execution_stall_score",
    "exp13_recovery_score",
    "exp13_compound_pressure_score",
    "exp13_pressure_imbalance",
    "exp13_revision_volatility_score",
]

CROSS_SIGNAL_FEATURES = [
    "exp13_cost_schedule_pressure_interaction",
    "exp13_cost_execution_interaction",
    "exp13_schedule_execution_interaction",
    "exp13_cost_schedule_acceleration_interaction",
    "exp13_worsening_streak_interaction",
]

TURNING_FEATURES = [
    "exp13_cost_short_long_divergence",
    "exp13_schedule_short_long_divergence",
    "exp13_spend_short_long_divergence",
    "exp13_cost_turning_strength",
    "exp13_schedule_turning_strength",
    "exp13_spend_turning_strength",
    "exp13_synchronized_turning_strength",
]

TRANSITION_FEATURES = [
    "exp13_cost_pressure_velocity_3m",
    "exp13_schedule_pressure_velocity_3m",
    "exp13_compound_pressure_velocity_3m",
    "exp13_worsening_transition_strength",
    "exp13_recovery_transition_strength",
    "exp13_worsening_regime_streak",
]

LIFECYCLE_FEATURES = [
    "exp13_lifecycle_progress",
    "exp13_cost_pressure_x_lifecycle",
    "exp13_schedule_pressure_x_lifecycle",
    "exp13_compound_pressure_x_lifecycle",
    "exp13_cost_growth_x_lifecycle",
    "exp13_slippage_acceleration_x_lifecycle",
    "exp13_spend_gap_x_lifecycle",
]

EXP13_FEATURES = list(
    dict.fromkeys(
        REGIME_FEATURES
        + CROSS_SIGNAL_FEATURES
        + TURNING_FEATURES
        + TRANSITION_FEATURES
        + LIFECYCLE_FEATURES
    )
)

_HISTORY_FEATURES = list(
    dict.fromkeys(REGIME_FEATURES + CROSS_SIGNAL_FEATURES + TURNING_FEATURES + TRANSITION_FEATURES)
)

_STAGE_PROGRESS = {
    "early": 0.125,
    "early_mid": 0.375,
    "mid": 0.375,
    "late_mid": 0.625,
    "late": 0.625,
    "very_late": 0.875,
}


def _array(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _positive_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    out[finite] = np.log1p(np.maximum(values[finite], 0.0))
    return out


def _weighted_available_score(terms: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Weighted mean of available, non-negative log-pressure terms."""
    if not terms:
        return np.array([], dtype=float)
    length = len(terms[0][0])
    numerator = np.zeros(length, dtype=float)
    denominator = np.zeros(length, dtype=float)
    for values, weight in terms:
        transformed = _positive_log(values)
        finite = np.isfinite(transformed)
        numerator[finite] += transformed[finite] * float(weight)
        denominator[finite] += float(weight)
    out = np.full(length, np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _binary_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    out = np.full(left.shape, np.nan, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    out[valid] = left[valid] * right[valid]
    return out


def _signed_log_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    out = np.full(left.shape, np.nan, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    product = left[valid] * right[valid]
    out[valid] = np.sign(product) * np.log1p(np.abs(left[valid])) * np.log1p(np.abs(right[valid]))
    return out


def _difference(short: np.ndarray, long: np.ndarray) -> np.ndarray:
    short = np.asarray(short, dtype=float)
    long = np.asarray(long, dtype=float)
    out = np.full(short.shape, np.nan, dtype=float)
    valid = np.isfinite(short) & np.isfinite(long)
    out[valid] = short[valid] - long[valid]
    return out


def _strength(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    out[finite] = np.log1p(np.abs(values[finite]))
    return out


def _positive_streak_from_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=float)
    streak = 0
    for i, value in enumerate(values):
        if np.isfinite(value) and value > 0:
            streak += 1
        elif np.isfinite(value):
            streak = 0
        out[i] = float(streak)
    return out


def engineer_history(history: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe regime, turning-point and transition context.

    Every feature at snapshot t is a deterministic function of snapshot t and
    earlier official reports for that canonical project.
    """
    frame = engineer_exp12_history(history)
    for feature in _HISTORY_FEATURES:
        frame[feature] = np.nan

    cost_v3 = _array(frame, "exp12_cost_growth_pct_3m")
    cost_v6 = _array(frame, "exp12_cost_growth_pct_6m")
    cost_v12 = _array(frame, "exp12_cost_growth_pct_12m")
    cost_accel = _array(frame, "exp12_cost_growth_pct_acceleration")
    cost_revision = _array(frame, "exp12_cost_revision_magnitude_12m_pct")
    cost_streak = _array(frame, "exp12_cost_worsening_streak")

    schedule_v3 = _array(frame, "exp12_slippage_ratio_velocity_3m")
    schedule_v6 = _array(frame, "exp12_slippage_ratio_velocity_6m")
    schedule_v12 = _array(frame, "exp12_slippage_ratio_velocity_12m")
    schedule_accel = _array(frame, "exp12_slippage_ratio_acceleration")
    schedule_revision = _array(frame, "exp12_schedule_revision_magnitude_12m_pct")
    schedule_streak = _array(frame, "exp12_slippage_worsening_streak")

    spend_v3 = _array(frame, "exp12_expenditure_ratio_velocity_3m")
    spend_v6 = _array(frame, "exp12_expenditure_ratio_velocity_6m")
    spend_v12 = _array(frame, "exp12_expenditure_ratio_velocity_12m")
    spend_accel = _array(frame, "exp12_expenditure_ratio_acceleration")
    spend_gap = _array(frame, "exp12_spend_vs_expected_progress_gap")

    cost_pressure = _weighted_available_score(
        [
            (cost_v3, 1.0),
            (cost_accel, 0.75),
            (cost_revision, 0.35),
            (cost_streak, 0.50),
        ]
    )
    schedule_pressure = _weighted_available_score(
        [
            (schedule_v3, 1.0),
            (schedule_accel, 0.75),
            (schedule_revision, 0.35),
            (schedule_streak, 0.50),
        ]
    )
    execution_stall = _weighted_available_score(
        [
            (-spend_v3, 1.0),
            (-spend_accel, 0.75),
            (-spend_gap, 0.50),
        ]
    )
    recovery = _weighted_available_score(
        [
            (-cost_v3, 1.0),
            (-schedule_v3, 1.0),
            (spend_accel, 0.35),
        ]
    )

    compound = np.full(len(frame), np.nan, dtype=float)
    valid_compound = np.isfinite(cost_pressure) & np.isfinite(schedule_pressure)
    compound[valid_compound] = np.sqrt(
        np.maximum(cost_pressure[valid_compound], 0.0)
        * np.maximum(schedule_pressure[valid_compound], 0.0)
    )
    imbalance = np.full(len(frame), np.nan, dtype=float)
    imbalance[valid_compound] = cost_pressure[valid_compound] - schedule_pressure[valid_compound]
    revision_volatility = _weighted_available_score(
        [(cost_revision, 1.0), (schedule_revision, 1.0)]
    )

    frame["exp13_cost_pressure_score"] = cost_pressure
    frame["exp13_schedule_pressure_score"] = schedule_pressure
    frame["exp13_execution_stall_score"] = execution_stall
    frame["exp13_recovery_score"] = recovery
    frame["exp13_compound_pressure_score"] = compound
    frame["exp13_pressure_imbalance"] = imbalance
    frame["exp13_revision_volatility_score"] = revision_volatility

    frame["exp13_cost_schedule_pressure_interaction"] = _binary_product(cost_pressure, schedule_pressure)
    frame["exp13_cost_execution_interaction"] = _binary_product(cost_pressure, execution_stall)
    frame["exp13_schedule_execution_interaction"] = _binary_product(schedule_pressure, execution_stall)
    frame["exp13_cost_schedule_acceleration_interaction"] = _signed_log_product(cost_accel, schedule_accel)
    frame["exp13_worsening_streak_interaction"] = _binary_product(
        _positive_log(cost_streak), _positive_log(schedule_streak)
    )

    cost_divergence = _difference(cost_v3, cost_v12)
    schedule_divergence = _difference(schedule_v3, schedule_v12)
    spend_divergence = _difference(spend_v3, spend_v12)
    frame["exp13_cost_short_long_divergence"] = cost_divergence
    frame["exp13_schedule_short_long_divergence"] = schedule_divergence
    frame["exp13_spend_short_long_divergence"] = spend_divergence
    frame["exp13_cost_turning_strength"] = _strength(cost_divergence)
    frame["exp13_schedule_turning_strength"] = _strength(schedule_divergence)
    frame["exp13_spend_turning_strength"] = _strength(spend_divergence)
    frame["exp13_synchronized_turning_strength"] = _signed_log_product(
        cost_divergence, schedule_divergence
    )

    # Transition features operate on the regime scores themselves.  _velocity
    # only looks backwards within the same project, so these remain as-of safe.
    for _, group in frame.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group.snapshot_date.astype("int64").to_numpy(np.int64)
        cp = pd.to_numeric(group.exp13_cost_pressure_score, errors="coerce").to_numpy(float)
        sp = pd.to_numeric(group.exp13_schedule_pressure_score, errors="coerce").to_numpy(float)
        xp = pd.to_numeric(group.exp13_compound_pressure_score, errors="coerce").to_numpy(float)
        cp_v = _velocity(dates, cp, WINDOW_DAYS[3])
        sp_v = _velocity(dates, sp, WINDOW_DAYS[3])
        xp_v = _velocity(dates, xp, WINDOW_DAYS[3])
        worsening = _weighted_available_score([(cp_v, 1.0), (sp_v, 1.0), (xp_v, 0.75)])
        recovering = _weighted_available_score([(-cp_v, 1.0), (-sp_v, 1.0), (-xp_v, 0.75)])
        frame.loc[idx, "exp13_cost_pressure_velocity_3m"] = cp_v
        frame.loc[idx, "exp13_schedule_pressure_velocity_3m"] = sp_v
        frame.loc[idx, "exp13_compound_pressure_velocity_3m"] = xp_v
        frame.loc[idx, "exp13_worsening_transition_strength"] = worsening
        frame.loc[idx, "exp13_recovery_transition_strength"] = recovering
        frame.loc[idx, "exp13_worsening_regime_streak"] = _positive_streak_from_signal(worsening)

    return frame


def _stage_progress(rows: pd.DataFrame) -> np.ndarray:
    if "lifecycle_stage" not in rows:
        return np.full(len(rows), np.nan, dtype=float)
    return rows["lifecycle_stage"].astype("string").str.lower().map(_STAGE_PROGRESS).to_numpy(dtype=float)


def enrich_rows(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach Exp13 context without changing the supervised cohort."""
    if history is None:
        if not TRAJECTORIES.exists():
            raise FileNotFoundError("Experiment 13 requires paimana_project_trajectories.csv.")
        history = pd.read_csv(
            TRAJECTORIES,
            dtype={"canonical_project_id": "string"},
            low_memory=False,
        )
    source = engineer_history(history)
    lookup = source[
        ["canonical_project_id", "snapshot_date", *_HISTORY_FEATURES, "exp12_history_12m"]
    ].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")

    rows = supervised.copy()
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    # Avoid duplicate history-depth columns when callers already enriched rows with
    # the promoted Exp12 production representation.
    if "exp12_history_12m" in rows:
        rows = rows.drop(columns=["exp12_history_12m"])
    result = rows.merge(
        lookup,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    if len(result) != len(supervised):
        raise AssertionError("Experiment 13 changed the supervised cohort.")

    progress = _stage_progress(result)
    cost_pressure = _array(result, "exp13_cost_pressure_score")
    schedule_pressure = _array(result, "exp13_schedule_pressure_score")
    compound_pressure = _array(result, "exp13_compound_pressure_score")
    cost_growth = _array(result, "exp12_cost_growth_pct_3m")
    schedule_accel = _array(result, "exp12_slippage_ratio_acceleration")
    spend_gap = _array(result, "exp12_spend_vs_expected_progress_gap")

    result["exp13_lifecycle_progress"] = progress
    result["exp13_cost_pressure_x_lifecycle"] = _binary_product(cost_pressure, progress)
    result["exp13_schedule_pressure_x_lifecycle"] = _binary_product(schedule_pressure, progress)
    result["exp13_compound_pressure_x_lifecycle"] = _binary_product(compound_pressure, progress)
    result["exp13_cost_growth_x_lifecycle"] = _binary_product(cost_growth, progress)
    result["exp13_slippage_acceleration_x_lifecycle"] = _binary_product(schedule_accel, progress)
    result["exp13_spend_gap_x_lifecycle"] = _binary_product(spend_gap, progress)
    return result


def _usable_features(train: pd.DataFrame) -> tuple[list[str], dict]:
    selected: list[str] = []
    audit: dict[str, dict] = {}
    for name in EXP13_FEATURES:
        values = pd.to_numeric(train.get(name), errors="coerce")
        if values is None:
            audit[name] = {"availability_percentage": 0.0, "selected": False}
            continue
        availability = float(values.notna().mean() * 100.0)
        usable = availability >= 10.0 and values.dropna().nunique() > 1
        audit[name] = {
            "availability_percentage": round(availability, 3),
            "selected": bool(usable),
        }
        if usable:
            selected.append(name)
    return selected, audit


def _candidate_groups(usable: list[str]) -> dict[str, list[str]]:
    allowed = set(usable)

    def keep(names: list[str]) -> list[str]:
        return [name for name in names if name in allowed]

    return {
        "production_only": [],
        "regime_scores": keep(REGIME_FEATURES),
        "regime_plus_interactions": keep(REGIME_FEATURES + CROSS_SIGNAL_FEATURES + LIFECYCLE_FEATURES),
        "regime_interactions_turning": keep(
            REGIME_FEATURES + CROSS_SIGNAL_FEATURES + LIFECYCLE_FEATURES + TURNING_FEATURES
        ),
        "all_regime_context": keep(EXP13_FEATURES),
    }


def _select_target_features(
    train: pd.DataFrame,
    production_features: list[str],
    usable: list[str],
    target: str,
    algorithm: str,
    seed: int,
) -> tuple[list[str], str, list[dict]]:
    """Select Exp13 feature depth strictly inside the training period."""
    years = sorted(
        pd.to_numeric(train.completion_year, errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    validation_years = years[-2:] if len(years) >= 3 else years[-1:]
    fitting = train[~train.completion_year.isin(validation_years)].copy()
    validation = train[train.completion_year.isin(validation_years)].copy()
    if fitting.canonical_project_id.nunique() < 5 or validation.canonical_project_id.nunique() < 2:
        fitting = train.copy()
        validation = train.copy()
        validation_years = years

    comparisons: list[dict] = []
    for group_name, added in _candidate_groups(usable).items():
        features = list(dict.fromkeys(production_features + added))
        model = _fit_pipeline(
            _regressors(seed)[algorithm],
            fitting,
            features,
            target,
        )
        prediction = model.predict(validation[features])
        metrics = _regression_metrics(
            validation[target],
            prediction,
            validation.sample_weight,
            validation.canonical_project_id,
        )
        comparisons.append(
            {
                "feature_group": group_name,
                "added_features": added,
                "feature_count": len(features),
                "validation_years": validation_years,
                **metrics,
            }
        )
    winner = min(comparisons, key=lambda item: item["MAE"])
    return list(winner["added_features"]), str(winner["feature_group"]), comparisons


def _percentage_gain(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return (float(baseline) - float(candidate)) / float(baseline) * 100.0


def fit_experiment(
    *,
    data,
    training_start,
    training_end,
    test_end,
    production_bundle,
    production_receipt,
    history=None,
):
    """Fit Exp13 against the current promoted Exp12 production baseline."""
    frozen = data.copy()
    frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce")
    frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(
        frozen, training_start, training_end, test_end
    )

    enriched = enrich_rows(frozen, history)
    train, test = temporal_project_split(
        enriched, training_start, training_end, test_end
    )
    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    production_cost_features = list(contract["cost"])
    production_delay_features = list(contract["delay"])
    if not production_cost_features or not production_delay_features:
        raise ValueError("Experiment 13 requires explicit production cost and delay feature contracts.")

    usable, availability_audit = _usable_features(train)
    if not usable:
        raise ValueError("Experiment 13 found no usable regime/context features in the training window.")

    cost_name = _algorithm(production_bundle, production_receipt, "cost")
    delay_name = _algorithm(production_bundle, production_receipt, "delay")
    cost_added, cost_group, cost_internal = _select_target_features(
        train,
        production_cost_features,
        usable,
        "actual_cost_overrun_percentage",
        cost_name,
        26303,
    )
    delay_added, delay_group, delay_internal = _select_target_features(
        train,
        production_delay_features,
        usable,
        "actual_delay_days",
        delay_name,
        26304,
    )
    cost_features = list(dict.fromkeys(production_cost_features + cost_added))
    delay_features = list(dict.fromkeys(production_delay_features + delay_added))

    cost_model = _fit_pipeline(
        _regressors(26303)[cost_name],
        train,
        cost_features,
        "actual_cost_overrun_percentage",
    )
    delay_model = _fit_pipeline(
        _regressors(26304)[delay_name],
        train,
        delay_features,
        "actual_delay_days",
    )

    compare = test[
        pd.to_numeric(test.exp12_history_12m, errors="coerce")
        .fillna(0)
        .ge(MIN_HISTORY)
    ].copy()
    if compare.canonical_project_id.nunique() < 2:
        raise ValueError("Experiment 13 has too few future projects with usable history.")
    compare = assign_project_balanced_weights(compare)

    compare["production_cost"] = production_bundle["cost"].predict(
        compare[production_cost_features]
    )
    compare["production_delay"] = np.maximum(
        0,
        production_bundle["delay"].predict(compare[production_delay_features]),
    )
    compare["experiment_cost"] = cost_model.predict(compare[cost_features])
    compare["experiment_delay"] = np.maximum(
        0,
        delay_model.predict(compare[delay_features]),
    )

    production_cost = _metric(
        compare, "actual_cost_overrun_percentage", "production_cost"
    )
    experiment_cost = _metric(
        compare, "actual_cost_overrun_percentage", "experiment_cost"
    )
    production_delay = _metric(compare, "actual_delay_days", "production_delay")
    experiment_delay = _metric(compare, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(
        compare,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost",
        candidate_prediction="experiment_cost",
        seed=26313,
    )
    paired_delay = paired_project_mae_comparison(
        compare,
        actual="actual_delay_days",
        baseline_prediction="production_delay",
        candidate_prediction="experiment_delay",
        seed=26314,
    )
    production_stage = _stage(compare, "production")
    experiment_stage = _stage(compare, "experiment")
    cost_gain = _percentage_gain(production_cost.get("MAE"), experiment_cost.get("MAE"))
    delay_gain = _percentage_gain(production_delay.get("MAE"), experiment_delay.get("MAE"))

    overall = {
        "production_cost_mae": production_cost["MAE"],
        "experiment_cost_mae": experiment_cost["MAE"],
        "absolute_mae_improvement_pp": round(
            production_cost["MAE"] - experiment_cost["MAE"], 4
        ),
        "improvement_percentage": round(cost_gain, 4) if cost_gain is not None else None,
        "production_delay_mae": production_delay["MAE"],
        "experiment_delay_mae": experiment_delay["MAE"],
        "absolute_delay_mae_improvement_days": round(
            production_delay["MAE"] - experiment_delay["MAE"], 4
        ),
        "delay_improvement_percentage": round(delay_gain, 4) if delay_gain is not None else None,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(compare)),
        "paired_project_comparison": paired_cost,
        "paired_project_cost_comparison": paired_cost,
        "paired_project_delay_comparison": paired_delay,
        "production_stage_metrics": production_stage,
        "experiment_stage_metrics": experiment_stage,
        "stage_balanced": {
            "production_cost_mae": _macro(production_stage, "cost"),
            "experiment_cost_mae": _macro(experiment_stage, "cost"),
            "production_delay_mae": _macro(production_stage, "delay"),
            "experiment_delay_mae": _macro(experiment_stage, "delay"),
        },
        "internal_feature_selection": {
            "cost": {"selected_group": cost_group, "comparisons": cost_internal},
            "delay": {"selected_group": delay_group, "comparisons": delay_internal},
        },
        "scientific_decision": "PENDING_TWO_WINDOW_AUDIT",
    }

    union_features = list(
        dict.fromkeys(cost_features + delay_features)
    )
    context = build_experiment_context(
        experiment_id=EXPERIMENT_ID,
        full_data=frozen,
        train=base_train,
        test=base_test,
        features=union_features,
        training_start=training_start,
        training_end=training_end,
        testing_end=test_end,
        weighting_policy="project-balanced quarterly snapshots",
    )
    leakage_policy = (
        "Exp13 first reproduces leakage-safe Exp12 trajectories, then derives regime, interaction, "
        "turning-point and transition features only from the current/earlier reports of the same "
        "canonical project. Feature-group selection uses only an internal historical block within "
        "the training period; the future holdout is never consulted for selection."
    )
    manifest = new_experiment_manifest(
        context=context,
        name=EXPERIMENT_NAME,
        changed_dimension="trajectory_regime_context",
        hypothesis=(
            "Conditioning trajectory signals on project deterioration/recovery state, lifecycle, "
            "cross-signal interactions and recent regime transitions improves future cost and/or "
            "delay MAE beyond the promoted Exp12 production cost baseline."
        ),
    )
    manifest.update(
        {
            "scope": EXPERIMENT_SCOPE,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "production_run_id": production_receipt.get("run_id"),
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "production_cost_features": production_cost_features,
            "production_delay_features": production_delay_features,
            "usable_exp13_features": usable,
            "feature_availability": availability_audit,
            "cost_added_features": cost_added,
            "delay_added_features": delay_added,
            "selected_feature_groups": {"cost": cost_group, "delay": delay_group},
            "internal_feature_selection": {
                "cost": cost_internal,
                "delay": delay_internal,
            },
            "selected_algorithms": {"cost": cost_name, "delay": delay_name},
            "comparison_filter": f">={MIN_HISTORY} official observations in trailing 12 months",
            "leakage_policy": leakage_policy,
            "promotion_rule": (
                "No automatic promotion. Cost and delay are judged independently on both 2001-2019 "
                "and 2001-2021 training windows with paired project evidence."
            ),
        }
    )

    run_dir = experiment_run_directory(
        EXPERIMENT_ID, context.window, manifest["run_id"]
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost_model, run_dir / "cost_model.pkl")
    joblib.dump(delay_model, run_dir / "delay_model.pkl")
    (run_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, allow_nan=False) + "\n"
    )
    (run_dir / "evaluation_results.json").write_text(
        json.dumps(_safe(overall), indent=2, allow_nan=False) + "\n"
    )
    record_experiment(
        {
            "experiment_id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "run_id": manifest["run_id"],
            "status": "COMPLETED",
            "decision": "PENDING",
            "model_role": "experiment",
            "promotion_allowed": False,
            "scope": EXPERIMENT_SCOPE,
            "window": context.window,
            "created_at": manifest["created_at"],
            "production_run_id": production_receipt.get("run_id"),
            "cost_improvement_percentage": overall["improvement_percentage"],
            "delay_improvement_percentage": overall["delay_improvement_percentage"],
            "implementation_revision": IMPLEMENTATION_REVISION,
        }
    )

    selected_added = list(dict.fromkeys(cost_added + delay_added))
    context_for_prediction = list(
        dict.fromkeys(
            selected_added
            + [
                "exp13_cost_pressure_score",
                "exp13_schedule_pressure_score",
                "exp13_compound_pressure_score",
                "exp13_worsening_transition_strength",
                "exp13_recovery_transition_strength",
            ]
        )
    )
    lookup_rows = enriched[
        ["canonical_project_id", "snapshot_date", *context_for_prediction]
    ].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    lookup = {
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): {
            feature: row.get(feature) for feature in context_for_prediction
        }
        for _, row in lookup_rows.iterrows()
        if pd.notna(row.canonical_project_id) and pd.notna(row.snapshot_date)
    }
    comparable = {
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
        for _, row in compare.iterrows()
    }

    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": manifest["run_id"],
        "model_role": "experiment",
        "scope": EXPERIMENT_SCOPE,
        "decision": "PENDING",
        "promotion_allowed": False,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "production_cost_baseline": metadata.get("production_cost_baseline"),
        "cost_feature_count": len(cost_features),
        "delay_feature_count": len(delay_features),
        "cost_added_features": cost_added,
        "delay_added_features": delay_added,
        "selected_feature_groups": {"cost": cost_group, "delay": delay_group},
        "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        "metrics": {"cost": experiment_cost, "delay": experiment_delay},
        "leakage_policy": leakage_policy,
    }
    return {
        "experiment": experiment,
        "overall_comparison": overall,
        "runtime_state": {
            "cost_model": cost_model,
            "delay_model": delay_model,
            "cost_features": cost_features,
            "delay_features": delay_features,
            "selected_added": selected_added,
            "lookup": lookup,
            "comparable": comparable,
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[
        frame.apply(lambda row: _key(row) in state["comparable"], axis=1)
    ].copy()


def _context_value(values: dict, name: str) -> float | None:
    value = values.get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if math.isfinite(number) else None


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 13 trajectory/regime context is available for this snapshot.")
    candidate = row.copy()
    values = state["lookup"][key]
    for name, value in values.items():
        candidate[name] = value

    cost_x = candidate.to_frame().T.reindex(columns=state["cost_features"])
    delay_x = candidate.to_frame().T.reindex(columns=state["delay_features"])
    return {
        "predicted_cost_overrun": round(
            float(state["cost_model"].predict(cost_x)[0]), 4
        ),
        "predicted_delay_days": round(
            max(0.0, float(state["delay_model"].predict(delay_x)[0])), 4
        ),
        "trajectory_features_available": int(
            sum(pd.notna(candidate.get(feature)) for feature in state["selected_added"])
        ),
        "trajectory_feature_count": len(state["selected_added"]),
        "regime_context": {
            "cost_pressure": _context_value(values, "exp13_cost_pressure_score"),
            "schedule_pressure": _context_value(values, "exp13_schedule_pressure_score"),
            "compound_pressure": _context_value(values, "exp13_compound_pressure_score"),
            "worsening_transition": _context_value(values, "exp13_worsening_transition_strength"),
            "recovery_transition": _context_value(values, "exp13_recovery_transition_strength"),
        },
        "implementation_revision": IMPLEMENTATION_REVISION,
    }
