"""Experiment 13 v2: learned trajectory regimes with robust temporal selection.

This revision addresses the failure mode observed in Exp13 v1:

* regimes are learned from leakage-safe Exp12 trajectory vectors instead of being
  hand-coded pressure formulas;
* structural changes are detected with an online, past-only CUSUM detector;
* candidate feature sets must generalize across multiple rolling temporal folds;
* challenger training and feature selection explicitly prioritize early/mid
  lifecycle snapshots while final reporting remains project-balanced.

Production remains unchanged.  The latent-regime encoder is fit only on the
training side of each fold and on the full training window for the final model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import (
    build_experiment_context,
    experiment_run_directory,
    new_experiment_manifest,
)
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.experiments.trajectory_exp12 import (
    MIN_HISTORY,
    _algorithm,
    _key,
    _macro,
    _metric,
    _safe,
    _stage,
)
from backend.app.ml.experiments.trajectory_exp12_v2 import engineer_history as engineer_exp12_history
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import target_feature_contract

EXPERIMENT_ID = "exp_13"
EXPERIMENT_NAME = "Advanced trajectory/regime interaction model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 13
IMPLEMENTATION_REVISION = "v2_learned_regimes_multifold_stage_priority"

# These are leakage-safe Exp12 signals.  The unsupervised regime model learns how
# they co-occur rather than assigning a manually designed pressure formula.
REGIME_INPUTS = [
    "exp12_cost_growth_pct_3m",
    "exp12_cost_growth_pct_6m",
    "exp12_cost_growth_pct_12m",
    "exp12_cost_growth_pct_acceleration",
    "exp12_expenditure_ratio_velocity_3m",
    "exp12_expenditure_ratio_velocity_6m",
    "exp12_expenditure_ratio_velocity_12m",
    "exp12_expenditure_ratio_acceleration",
    "exp12_slippage_ratio_velocity_3m",
    "exp12_slippage_ratio_velocity_6m",
    "exp12_slippage_ratio_velocity_12m",
    "exp12_slippage_ratio_acceleration",
    "exp12_spend_vs_expected_progress_gap",
    "exp12_cost_revision_magnitude_12m_pct",
    "exp12_schedule_revision_magnitude_12m_pct",
    "exp12_cost_worsening_streak",
    "exp12_slippage_worsening_streak",
]

REGIME_FEATURES = [
    "exp13v2_regime_probability_0",
    "exp13v2_regime_probability_1",
    "exp13v2_regime_probability_2",
    "exp13v2_regime_probability_3",
    "exp13v2_regime_confidence",
    "exp13v2_regime_entropy",
    "exp13v2_regime_surprise",
    "exp13v2_regime_id",
]

CHANGE_POINT_FEATURES = [
    "exp13v2_cost_change_score",
    "exp13v2_cost_change_event",
    "exp13v2_cost_reports_since_change",
    "exp13v2_schedule_change_score",
    "exp13v2_schedule_change_event",
    "exp13v2_schedule_reports_since_change",
    "exp13v2_spend_change_score",
    "exp13v2_spend_change_event",
    "exp13v2_spend_reports_since_change",
    "exp13v2_synchronous_change_score",
]

EXP13_FEATURES = REGIME_FEATURES + CHANGE_POINT_FEATURES

_STAGE_TRAIN_MULTIPLIER = {
    "early": 2.5,
    "early_mid": 2.1,
    "mid": 2.1,
    "late_mid": 1.25,
    "late": 1.25,
    "very_late": 0.70,
}
_STAGE_OBJECTIVE_WEIGHT = {
    "early": 3.0,
    "early_mid": 2.5,
    "mid": 2.5,
    "late_mid": 1.25,
    "late": 1.25,
    "very_late": 0.50,
}


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _online_cusum(values: np.ndarray, *, min_history: int = 4, drift: float = 0.5, threshold: float = 4.0):
    """Past-only online two-sided CUSUM.

    The baseline for observation t uses only earlier observations from that
    project.  A detected event resets the cumulative statistic so repeated
    structural changes can be represented without peeking forward.
    """
    values = np.asarray(values, dtype=float)
    score = np.zeros(len(values), dtype=float)
    event = np.zeros(len(values), dtype=float)
    since = np.full(len(values), np.nan, dtype=float)
    history: list[float] = []
    positive = 0.0
    negative = 0.0
    last_event: int | None = None
    for i, value in enumerate(values):
        if np.isfinite(value) and len(history) >= min_history:
            baseline = np.asarray(history[-12:], dtype=float)
            centre = float(np.nanmedian(baseline))
            scale = float(np.nanstd(baseline, ddof=1))
            if not math.isfinite(scale) or scale < 1e-6:
                scale = max(float(np.nanmedian(np.abs(baseline - centre))) * 1.4826, 1e-3)
            z = (float(value) - centre) / scale
            positive = max(0.0, positive + z - drift)
            negative = min(0.0, negative + z + drift)
            current = max(positive, -negative)
            score[i] = current
            if current >= threshold:
                event[i] = 1.0
                last_event = i
                positive = 0.0
                negative = 0.0
        if last_event is not None:
            since[i] = float(i - last_event)
        if np.isfinite(value):
            history.append(float(value))
    return score, event, since


def engineer_change_points(history: pd.DataFrame) -> pd.DataFrame:
    """Attach online change-point features to leakage-safe Exp12 history."""
    frame = engineer_exp12_history(history).copy()
    for name in CHANGE_POINT_FEATURES:
        frame[name] = np.nan
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.sort_values(["canonical_project_id", "snapshot_date"]).copy()
    signal_map = {
        "cost": "exp12_cost_growth_pct_3m",
        "schedule": "exp12_slippage_ratio_velocity_3m",
        "spend": "exp12_expenditure_ratio_velocity_3m",
    }
    for _, group in frame.groupby("canonical_project_id", sort=False):
        idx = group.index
        scores: dict[str, np.ndarray] = {}
        for label, column in signal_map.items():
            score, event, since = _online_cusum(_numeric(group, column))
            frame.loc[idx, f"exp13v2_{label}_change_score"] = score
            frame.loc[idx, f"exp13v2_{label}_change_event"] = event
            frame.loc[idx, f"exp13v2_{label}_reports_since_change"] = since
            scores[label] = score
        stacked = np.vstack([scores["cost"], scores["schedule"], scores["spend"]])
        frame.loc[idx, "exp13v2_synchronous_change_score"] = np.nanmean(stacked, axis=0)
    return frame


def enrich_base_rows(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge raw leakage-safe trajectory inputs and CUSUM context onto supervised rows."""
    if history is None:
        if not TRAJECTORIES.exists():
            raise FileNotFoundError("Experiment 13 requires paimana_project_trajectories.csv.")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    source = engineer_change_points(history)
    available_inputs = [name for name in REGIME_INPUTS if name in source.columns]
    lookup_columns = [
        "canonical_project_id",
        "snapshot_date",
        *available_inputs,
        *CHANGE_POINT_FEATURES,
        "exp12_history_12m",
    ]
    lookup = source[lookup_columns].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )
    rows = supervised.copy()
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    drop_existing = [name for name in lookup_columns if name not in {"canonical_project_id", "snapshot_date"} and name in rows]
    if drop_existing:
        rows = rows.drop(columns=drop_existing)
    result = rows.merge(
        lookup,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    if len(result) != len(supervised):
        raise AssertionError("Experiment 13 v2 changed the supervised cohort.")
    return result


@dataclass
class LearnedRegimeEncoder:
    features: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    mixture: GaussianMixture


def fit_regime_encoder(train: pd.DataFrame, seed: int = 26313) -> LearnedRegimeEncoder:
    usable: list[str] = []
    for name in REGIME_INPUTS:
        if name not in train:
            continue
        values = pd.to_numeric(train[name], errors="coerce")
        if values.notna().mean() >= 0.10 and values.dropna().nunique() > 1:
            usable.append(name)
    if len(usable) < 3:
        raise ValueError("Experiment 13 v2 has too few usable trajectory dimensions for learned regimes.")
    matrix = train[usable].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(matrix))
    components = 4 if len(x) >= 16 else max(2, min(4, len(x) // 4))
    mixture = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        reg_covar=1e-4,
        n_init=3,
        max_iter=300,
        random_state=seed,
    )
    mixture.fit(x)
    return LearnedRegimeEncoder(usable, imputer, scaler, mixture)


def apply_regime_encoder(frame: pd.DataFrame, encoder: LearnedRegimeEncoder) -> pd.DataFrame:
    result = frame.copy()
    matrix = result[encoder.features].apply(pd.to_numeric, errors="coerce")
    x = encoder.scaler.transform(encoder.imputer.transform(matrix))
    probabilities = encoder.mixture.predict_proba(x)
    ids = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    surprise = -encoder.mixture.score_samples(x)
    for i in range(4):
        result[f"exp13v2_regime_probability_{i}"] = probabilities[:, i] if i < probabilities.shape[1] else 0.0
    result["exp13v2_regime_confidence"] = confidence
    result["exp13v2_regime_entropy"] = entropy
    result["exp13v2_regime_surprise"] = surprise
    result["exp13v2_regime_id"] = ids.astype(float)
    return result


def stage_aware_training_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Prioritize decision-useful early/mid snapshots during challenger fitting."""
    result = frame.copy()
    base = pd.to_numeric(result.get("sample_weight", 1.0), errors="coerce").fillna(0.0).to_numpy(float)
    if "lifecycle_stage" in result:
        stage = result.lifecycle_stage.astype("string").str.lower()
        multiplier = stage.map(_STAGE_TRAIN_MULTIPLIER).fillna(1.0).to_numpy(float)
    else:
        multiplier = np.ones(len(result), dtype=float)
    adjusted = base * multiplier
    base_mean = float(np.mean(base)) if len(base) else 1.0
    adjusted_mean = float(np.mean(adjusted)) if len(adjusted) else 1.0
    if adjusted_mean > 0:
        adjusted *= base_mean / adjusted_mean
    result["sample_weight"] = adjusted
    return result


def _validation_score(frame: pd.DataFrame, target: str, prediction: np.ndarray) -> dict:
    work = frame.copy()
    work["_prediction"] = np.asarray(prediction, dtype=float)
    overall = _regression_metrics(
        work[target], work["_prediction"], work.sample_weight, work.canonical_project_id
    )
    stage_mae: dict[str, float] = {}
    numerator = 0.0
    denominator = 0.0
    if "lifecycle_stage" in work:
        for stage_name, group in work.groupby(work.lifecycle_stage.astype("string").str.lower()):
            actual = pd.to_numeric(group[target], errors="coerce")
            pred = pd.to_numeric(group["_prediction"], errors="coerce")
            weight = pd.to_numeric(group.sample_weight, errors="coerce").fillna(0.0)
            valid = actual.notna() & pred.notna() & weight.gt(0)
            if not valid.any():
                continue
            mae = float(np.average(np.abs(actual[valid] - pred[valid]), weights=weight[valid]))
            stage_mae[str(stage_name)] = mae
            priority = float(_STAGE_OBJECTIVE_WEIGHT.get(str(stage_name), 1.0))
            numerator += priority * mae
            denominator += priority
    priority_mae = numerator / denominator if denominator else float(overall["MAE"])
    objective = 0.40 * float(overall["MAE"]) + 0.60 * priority_mae
    return {
        "MAE": float(overall["MAE"]),
        "stage_priority_mae": round(priority_mae, 6),
        "selection_objective": round(objective, 6),
        "stage_mae": {key: round(value, 6) for key, value in stage_mae.items()},
    }


def rolling_temporal_folds(train: pd.DataFrame, max_folds: int = 3) -> list[tuple[list[int], list[int]]]:
    years = sorted(pd.to_numeric(train.completion_year, errors="coerce").dropna().astype(int).unique())
    if len(years) < 4:
        return [(years[:-1], years[-1:])] if len(years) >= 2 else [(years, years)]
    width = 2 if len(years) >= 7 else 1
    folds: list[tuple[list[int], list[int]]] = []
    cursor = len(years)
    while cursor > 1 and len(folds) < max_folds:
        start = max(1, cursor - width)
        fit_years = years[:start]
        validation_years = years[start:cursor]
        fitting = train[train.completion_year.isin(fit_years)]
        validation = train[train.completion_year.isin(validation_years)]
        if fitting.canonical_project_id.nunique() >= 5 and validation.canonical_project_id.nunique() >= 2:
            folds.append((fit_years, validation_years))
        cursor = start
    folds.reverse()
    if not folds:
        return [(years[:-1], years[-1:])]
    return folds


def _candidate_groups() -> dict[str, list[str]]:
    return {
        "stage_weighted_production": [],
        "learned_regimes": list(REGIME_FEATURES),
        "learned_regimes_plus_change_points": list(REGIME_FEATURES + CHANGE_POINT_FEATURES),
    }


def _percentage_gain(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return (float(baseline) - float(candidate)) / float(baseline) * 100.0


def _select_target_features(
    train: pd.DataFrame,
    production_features: list[str],
    target: str,
    algorithm: str,
    seed: int,
) -> tuple[list[str], str, dict]:
    """Require a learned-regime group to generalize across rolling historical folds."""
    folds = rolling_temporal_folds(train, max_folds=3)
    group_records: dict[str, list[dict]] = {name: [] for name in _candidate_groups()}
    for fold_index, (fit_years, validation_years) in enumerate(folds):
        fitting_raw = train[train.completion_year.isin(fit_years)].copy()
        validation_raw = train[train.completion_year.isin(validation_years)].copy()
        encoder = fit_regime_encoder(fitting_raw, seed + fold_index)
        fitting = apply_regime_encoder(fitting_raw, encoder)
        validation = apply_regime_encoder(validation_raw, encoder)
        fitting = stage_aware_training_weights(fitting)
        # Evaluation weights stay project-balanced; only the fitting objective is stage-aware.
        for group_name, added in _candidate_groups().items():
            features = list(dict.fromkeys(production_features + added))
            model = _fit_pipeline(_regressors(seed + fold_index)[algorithm], fitting, features, target)
            prediction = model.predict(validation[features])
            score = _validation_score(validation, target, prediction)
            group_records[group_name].append(
                {
                    "fold": fold_index + 1,
                    "fit_years": fit_years,
                    "validation_years": validation_years,
                    **score,
                }
            )

    summary: dict[str, dict] = {}
    for group_name, records in group_records.items():
        summary[group_name] = {
            "mean_MAE": round(float(np.mean([item["MAE"] for item in records])), 6),
            "mean_stage_priority_mae": round(float(np.mean([item["stage_priority_mae"] for item in records])), 6),
            "mean_selection_objective": round(float(np.mean([item["selection_objective"] for item in records])), 6),
            "folds": records,
        }

    baseline_name = "stage_weighted_production"
    baseline_records = group_records[baseline_name]
    eligible: list[tuple[str, float, int, float]] = []
    for group_name, records in group_records.items():
        if group_name == baseline_name:
            continue
        fold_gains = [
            _percentage_gain(base["selection_objective"], cand["selection_objective"]) or 0.0
            for base, cand in zip(baseline_records, records)
        ]
        mean_gain = float(np.mean(fold_gains))
        wins = int(sum(gain > 0 for gain in fold_gains))
        worst = float(min(fold_gains))
        summary[group_name]["objective_gain_vs_stage_weighted_production_pct"] = round(mean_gain, 6)
        summary[group_name]["winning_folds"] = wins
        summary[group_name]["worst_fold_gain_pct"] = round(worst, 6)
        # Require a repeatable gain, not a one-window lucky result.
        if mean_gain >= 0.25 and wins >= max(2, len(folds) - 1) and worst > -1.5:
            eligible.append((group_name, mean_gain, wins, worst))

    if eligible:
        winner_name = max(eligible, key=lambda item: item[1])[0]
    else:
        winner_name = baseline_name
    added = _candidate_groups()[winner_name]
    audit = {
        "fold_count": len(folds),
        "selection_rule": (
            "learned regime features require >=0.25% mean stage-priority objective gain, wins on at least "
            "two rolling folds (or all-but-one when fewer exist), and no fold worse than -1.5%"
        ),
        "selected_group": winner_name,
        "groups": summary,
    }
    return list(added), winner_name, audit


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
    frozen = data.copy()
    frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce")
    frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(frozen, training_start, training_end, test_end)

    enriched_base = enrich_base_rows(frozen, history)
    train_raw, test_raw = temporal_project_split(enriched_base, training_start, training_end, test_end)
    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    production_cost_features = list(contract["cost"])
    production_delay_features = list(contract["delay"])
    if not production_cost_features or not production_delay_features:
        raise ValueError("Experiment 13 v2 requires explicit production cost and delay feature contracts.")

    cost_name = _algorithm(production_bundle, production_receipt, "cost")
    delay_name = _algorithm(production_bundle, production_receipt, "delay")
    cost_added, cost_group, cost_internal = _select_target_features(
        train_raw, production_cost_features, "actual_cost_overrun_percentage", cost_name, 27303
    )
    delay_added, delay_group, delay_internal = _select_target_features(
        train_raw, production_delay_features, "actual_delay_days", delay_name, 27304
    )

    final_encoder = fit_regime_encoder(train_raw, 27313)
    enriched = apply_regime_encoder(enriched_base, final_encoder)
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    fit_train = stage_aware_training_weights(train)
    cost_features = list(dict.fromkeys(production_cost_features + cost_added))
    delay_features = list(dict.fromkeys(production_delay_features + delay_added))
    cost_model = _fit_pipeline(
        _regressors(27303)[cost_name], fit_train, cost_features, "actual_cost_overrun_percentage"
    )
    delay_model = _fit_pipeline(
        _regressors(27304)[delay_name], fit_train, delay_features, "actual_delay_days"
    )

    compare = test[
        pd.to_numeric(test.exp12_history_12m, errors="coerce").fillna(0).ge(MIN_HISTORY)
    ].copy()
    if compare.canonical_project_id.nunique() < 2:
        raise ValueError("Experiment 13 v2 has too few future projects with usable history.")
    compare = assign_project_balanced_weights(compare)
    compare["production_cost"] = production_bundle["cost"].predict(compare[production_cost_features])
    compare["production_delay"] = np.maximum(
        0, production_bundle["delay"].predict(compare[production_delay_features])
    )
    compare["experiment_cost"] = cost_model.predict(compare[cost_features])
    compare["experiment_delay"] = np.maximum(0, delay_model.predict(compare[delay_features]))

    production_cost = _metric(compare, "actual_cost_overrun_percentage", "production_cost")
    experiment_cost = _metric(compare, "actual_cost_overrun_percentage", "experiment_cost")
    production_delay = _metric(compare, "actual_delay_days", "production_delay")
    experiment_delay = _metric(compare, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(
        compare,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost",
        candidate_prediction="experiment_cost",
        seed=27313,
    )
    paired_delay = paired_project_mae_comparison(
        compare,
        actual="actual_delay_days",
        baseline_prediction="production_delay",
        candidate_prediction="experiment_delay",
        seed=27314,
    )
    production_stage = _stage(compare, "production")
    experiment_stage = _stage(compare, "experiment")
    cost_gain = _percentage_gain(production_cost.get("MAE"), experiment_cost.get("MAE"))
    delay_gain = _percentage_gain(production_delay.get("MAE"), experiment_delay.get("MAE"))

    overall = {
        "production_cost_mae": production_cost["MAE"],
        "experiment_cost_mae": experiment_cost["MAE"],
        "absolute_mae_improvement_pp": round(production_cost["MAE"] - experiment_cost["MAE"], 4),
        "improvement_percentage": round(cost_gain, 4) if cost_gain is not None else None,
        "production_delay_mae": production_delay["MAE"],
        "experiment_delay_mae": experiment_delay["MAE"],
        "absolute_delay_mae_improvement_days": round(production_delay["MAE"] - experiment_delay["MAE"], 4),
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
            "cost": cost_internal,
            "delay": delay_internal,
        },
        "scientific_decision": "PENDING_TWO_WINDOW_AUDIT",
    }

    union_features = list(dict.fromkeys(cost_features + delay_features))
    context = build_experiment_context(
        experiment_id=EXPERIMENT_ID,
        full_data=frozen,
        train=base_train,
        test=base_test,
        features=union_features,
        training_start=training_start,
        training_end=training_end,
        testing_end=test_end,
        weighting_policy=(
            "stage-aware challenger training (early/mid prioritized); project-balanced quarterly evaluation"
        ),
    )
    leakage_policy = (
        "Exp13 v2 derives Exp12 trajectories and online CUSUM change scores from current/earlier reports only. "
        "Each rolling validation fold fits its latent Gaussian-mixture regime encoder only on that fold's "
        "historical fitting years. The final encoder is fit only on the full training window; future holdout "
        "projects are transformed but never used to learn regimes or choose feature groups."
    )
    manifest = new_experiment_manifest(
        context=context,
        name=EXPERIMENT_NAME,
        changed_dimension="trajectory_regime_context",
        hypothesis=(
            "Learned latent trajectory states plus past-only change points, selected across rolling temporal "
            "folds and trained with early/mid lifecycle priority, improve future cost and/or delay MAE beyond Exp12."
        ),
    )
    manifest.update(
        {
            "scope": EXPERIMENT_SCOPE,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "production_run_id": production_receipt.get("run_id"),
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "regime_input_features": final_encoder.features,
            "regime_components": int(final_encoder.mixture.n_components),
            "cost_added_features": cost_added,
            "delay_added_features": delay_added,
            "selected_feature_groups": {"cost": cost_group, "delay": delay_group},
            "internal_feature_selection": {"cost": cost_internal, "delay": delay_internal},
            "selected_algorithms": {"cost": cost_name, "delay": delay_name},
            "stage_training_multipliers": _STAGE_TRAIN_MULTIPLIER,
            "stage_selection_weights": _STAGE_OBJECTIVE_WEIGHT,
            "comparison_filter": f">={MIN_HISTORY} official observations in trailing 12 months",
            "leakage_policy": leakage_policy,
            "promotion_rule": (
                "No automatic promotion. Cost and delay must independently improve on both 2001-2019 and "
                "2001-2021 future holdouts, with paired-project and lifecycle-stage evidence."
            ),
        }
    )

    run_dir = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"])
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost_model, run_dir / "cost_model.pkl")
    joblib.dump(delay_model, run_dir / "delay_model.pkl")
    joblib.dump(final_encoder, run_dir / "learned_regime_encoder.pkl")
    (run_dir / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, allow_nan=False) + "\n")
    (run_dir / "evaluation_results.json").write_text(json.dumps(_safe(overall), indent=2, allow_nan=False) + "\n")
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
    context_for_prediction = list(dict.fromkeys(selected_added + REGIME_FEATURES + CHANGE_POINT_FEATURES))
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
            "regime_encoder": final_encoder,
            "lookup": lookup,
            "comparable": comparable,
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


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
        raise ValueError("No Experiment 13 v2 learned-regime context is available for this snapshot.")
    candidate = row.copy()
    values = state["lookup"][key]
    for name, value in values.items():
        candidate[name] = value
    cost_x = candidate.to_frame().T.reindex(columns=state["cost_features"])
    delay_x = candidate.to_frame().T.reindex(columns=state["delay_features"])
    return {
        "predicted_cost_overrun": round(float(state["cost_model"].predict(cost_x)[0]), 4),
        "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(delay_x)[0])), 4),
        "trajectory_features_available": int(sum(pd.notna(candidate.get(feature)) for feature in state["selected_added"])),
        "trajectory_feature_count": len(state["selected_added"]),
        "regime_context": {
            "regime_id": _context_value(values, "exp13v2_regime_id"),
            "confidence": _context_value(values, "exp13v2_regime_confidence"),
            "entropy": _context_value(values, "exp13v2_regime_entropy"),
            "surprise": _context_value(values, "exp13v2_regime_surprise"),
            "cost_change_score": _context_value(values, "exp13v2_cost_change_score"),
            "schedule_change_score": _context_value(values, "exp13v2_schedule_change_score"),
            "spend_change_score": _context_value(values, "exp13v2_spend_change_score"),
        },
        "implementation_revision": IMPLEMENTATION_REVISION,
    }
