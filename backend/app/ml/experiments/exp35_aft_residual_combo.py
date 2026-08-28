"""Experiment 35: combine Exp32 AFT remaining-time forecasting with Exp33 residual calibration.

Current production is Exp12 Cost + Exp34 Delay.

Cost:
- Exp32 is Delay-only, so Cost changes only through Exp33's cross-fitted
  weighted-median residual calibration on the current Exp12 Cost model family.

Delay:
- Keep the current Exp34 feature contract and three-family blend weights fixed.
- Replace the direct delay target with Exp32's log1p(remaining-days) target.
- Convert predicted remaining time back to final delay against the as-of planned
  completion date.
- Apply Exp33's cross-fitted stage/prediction-bin weighted-median residual
  correction to those AFT delay predictions.

All calibration data comes from rolling validation years inside the training
window. The future holdout is never used for model, weight, or calibration
selection.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    FAMILIES,
    _rolling_folds,
    enrich_path_dependence,
)
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
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
    target_feature_contract,
)

EXPERIMENT_ID = "exp_35"
EXPERIMENT_NAME = "Exp32 AFT remaining-time + Exp33 cross-fitted residual calibration"
EXPERIMENT_SCOPE = "cost+delay"
EXPERIMENT_SEQUENCE = 35
DELAY_SEED = 26204
N_BINS = 5
MIN_GROUP_ROWS = 20


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _weighted_median(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = 0.5 * weights.sum()
    if cutoff <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _fit_residual_calibration(oof: pd.DataFrame) -> dict:
    finite = np.isfinite(pd.to_numeric(oof["prediction"], errors="coerce").to_numpy(float))
    if finite.sum() < 20:
        raise ValueError("Exp35 has insufficient finite OOF predictions for calibration.")

    predictions = pd.to_numeric(oof.loc[finite, "prediction"], errors="coerce")
    edges = np.unique(
        np.quantile(predictions, np.linspace(0, 1, N_BINS + 1)).astype(float)
    )
    if len(edges) < 3:
        edges = np.array([-np.inf, float(np.median(predictions)), np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    work = oof.copy()
    work["bin"] = np.digitize(
        pd.to_numeric(work["prediction"], errors="coerce").to_numpy(float),
        edges[1:-1],
        right=False,
    )
    global_median = _weighted_median(work["residual"], work["sample_weight"])
    bin_medians = {
        int(b): _weighted_median(part["residual"], part["sample_weight"])
        for b, part in work.groupby("bin")
    }
    stage_bin_medians = {}
    for (stage, b), part in work.groupby(["lifecycle_stage", "bin"], dropna=False):
        if len(part) < MIN_GROUP_ROWS:
            continue
        stage_key = "<NA>" if pd.isna(stage) else str(stage)
        stage_bin_medians[(stage_key, int(b))] = _weighted_median(
            part["residual"], part["sample_weight"]
        )
    return {
        "edges": edges.tolist(),
        "global_median": float(global_median),
        "bin_medians": bin_medians,
        "stage_bin_medians": stage_bin_medians,
        "oof_rows": int(len(work)),
    }


def _public_calibration(calibration: dict) -> dict:
    edges = []
    for value in calibration["edges"]:
        numeric = float(value)
        edges.append(numeric if np.isfinite(numeric) else None)
    return {
        "edges": edges,
        "edge_semantics": "null first/last edge means unbounded",
        "global_median": float(calibration["global_median"]),
        "oof_rows": int(calibration["oof_rows"]),
    }


def _corrections(
    frame: pd.DataFrame, predictions: np.ndarray, calibration: dict
) -> np.ndarray:
    edges = np.asarray(calibration["edges"], dtype=float)
    bins = np.digitize(np.asarray(predictions, dtype=float), edges[1:-1], right=False)
    result = np.zeros(len(frame), dtype=float)
    stages = frame.get("lifecycle_stage", pd.Series(pd.NA, index=frame.index))
    for i, (stage, b) in enumerate(zip(stages, bins)):
        stage_key = "<NA>" if pd.isna(stage) else str(stage)
        correction = calibration["stage_bin_medians"].get((stage_key, int(b)))
        if correction is None:
            correction = calibration["bin_medians"].get(
                int(b), calibration["global_median"]
            )
        result[i] = float(correction)
    return result


def _cost_calibration_oof(
    train: pd.DataFrame, features: list[str], algorithm: str
) -> tuple[dict, list[dict]]:
    folds = _rolling_folds(train)
    if len(folds) < 2:
        raise ValueError("Exp35 Cost calibration requires at least two rolling folds.")
    chunks = []
    diagnostics = []
    for fitting, validation, year in folds:
        model = _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[algorithm],
            fitting,
            features,
            "actual_cost_overrun_percentage",
        )
        pred = model.predict(validation[features])
        chunk = validation[
            [
                "actual_cost_overrun_percentage",
                "sample_weight",
                "canonical_project_id",
                "lifecycle_stage",
            ]
        ].copy()
        chunk["prediction"] = pred
        chunk["residual"] = (
            pd.to_numeric(chunk["actual_cost_overrun_percentage"], errors="coerce")
            - pred
        )
        chunks.append(chunk)
        diagnostics.append({
            "year": int(year),
            "projects": int(validation.canonical_project_id.nunique()),
            "MAE": _regression_metrics(
                validation.actual_cost_overrun_percentage,
                pred,
                validation.sample_weight,
                validation.canonical_project_id,
            )["MAE"],
        })
    oof = pd.concat(chunks, ignore_index=True)
    return _fit_residual_calibration(oof), diagnostics


def _remaining_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["completion_date"] = pd.to_datetime(out.get("completion_date"), errors="coerce")
    out["planned_completion_date"] = pd.to_datetime(
        out.get("planned_completion_date"), errors="coerce"
    )
    remaining = (out["completion_date"] - out["snapshot_date"]).dt.days
    mask = remaining.gt(0) & out["planned_completion_date"].notna()
    out = out[mask].copy()
    out["exp35_remaining_days"] = remaining[mask].astype(float)
    out["exp35_log_remaining_days"] = np.log1p(out["exp35_remaining_days"])
    return assign_project_balanced_weights(out)


def _delay_from_remaining(
    frame: pd.DataFrame, remaining_days: np.ndarray
) -> np.ndarray:
    remaining = np.maximum(0.0, np.asarray(remaining_days, dtype=float))
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    planned = pd.to_datetime(frame["planned_completion_date"], errors="coerce")
    predicted_completion = snapshot + pd.to_timedelta(remaining, unit="D")
    delay = (
        (predicted_completion - planned).dt.total_seconds().to_numpy(dtype=float)
        / 86400.0
    )
    return np.maximum(0.0, delay)


def _fit_aft_family_models(
    train: pd.DataFrame, features: list[str]
) -> dict[str, object]:
    return {
        family: _fit_pipeline(
            _regressors(DELAY_SEED)[family],
            train,
            features,
            "exp35_log_remaining_days",
        )
        for family in FAMILIES
    }


def _aft_remaining_prediction(
    models: dict[str, object],
    weights: dict[str, float],
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    log_prediction = np.zeros(len(frame), dtype=float)
    for family in FAMILIES:
        log_prediction += float(weights[family]) * models[family].predict(
            frame[features]
        )
    return np.maximum(0.0, np.expm1(np.clip(log_prediction, -20, 20)))


def _delay_aft_calibration_oof(
    train_delay: pd.DataFrame,
    features: list[str],
    weights: dict[str, float],
) -> tuple[dict, list[dict]]:
    folds = _rolling_folds(train_delay)
    if len(folds) < 2:
        raise ValueError("Exp35 Delay calibration requires at least two rolling folds.")
    chunks = []
    diagnostics = []
    for fitting, validation, year in folds:
        models = _fit_aft_family_models(fitting, features)
        remaining = _aft_remaining_prediction(models, weights, validation, features)
        pred = _delay_from_remaining(validation, remaining)
        chunk = validation[
            [
                "actual_delay_days",
                "sample_weight",
                "canonical_project_id",
                "lifecycle_stage",
            ]
        ].copy()
        chunk["prediction"] = pred
        chunk["residual"] = pd.to_numeric(
            chunk["actual_delay_days"], errors="coerce"
        ) - pred
        chunks.append(chunk)
        diagnostics.append({
            "year": int(year),
            "projects": int(validation.canonical_project_id.nunique()),
            "MAE_before_calibration": _regression_metrics(
                validation.actual_delay_days,
                pred,
                validation.sample_weight,
                validation.canonical_project_id,
            )["MAE"],
        })
    oof = pd.concat(chunks, ignore_index=True)
    return _fit_residual_calibration(oof), diagnostics


def fit_experiment(
    *,
    data,
    training_start,
    training_end,
    test_end,
    production_bundle,
    production_receipt,
    **_,
):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(
        enriched["completion_year"], errors="coerce"
    )
    enriched["snapshot_date"] = pd.to_datetime(
        enriched["snapshot_date"], errors="coerce"
    )
    train, test = temporal_project_split(
        enriched, training_start, training_end, test_end
    )

    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    cost_features = list(contract["cost"])
    delay_features = list(contract["delay"])
    selected = dict(
        metadata.get("selected_algorithms")
        or production_receipt.get("selected_algorithms")
        or {}
    )
    cost_algorithm = selected.get("cost")
    if cost_algorithm not in _regressors(PRODUCTION_COST_SEED):
        raise ValueError(
            f"Exp35 requires a standard current production Cost family; got {cost_algorithm!r}."
        )

    delay_weights = {
        family: float((metadata.get("delay_blend_weights") or {}).get(family, 0.0))
        for family in FAMILIES
    }
    if abs(sum(delay_weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"Exp35 requires normalized current Exp34 Delay weights; got {delay_weights}."
        )

    cost_calibration, cost_oof = _cost_calibration_oof(
        train, cost_features, cost_algorithm
    )

    train_delay = _remaining_frame(train)
    delay_calibration, delay_oof = _delay_aft_calibration_oof(
        train_delay, delay_features, delay_weights
    )
    aft_models = _fit_aft_family_models(train_delay, delay_features)

    cost_compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(cost_compare[cost_features])
    exp_cost_pred = prod_cost_pred + _corrections(
        cost_compare, prod_cost_pred, cost_calibration
    )
    prod_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage,
        prod_cost_pred,
        cost_compare.sample_weight,
        cost_compare.canonical_project_id,
    )
    exp_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage,
        exp_cost_pred,
        cost_compare.sample_weight,
        cost_compare.canonical_project_id,
    )

    shared_delay_full_pred = np.maximum(
        0.0, production_bundle["delay"].predict(cost_compare[delay_features])
    )
    shared_delay_full = _regression_metrics(
        cost_compare.actual_delay_days,
        shared_delay_full_pred,
        cost_compare.sample_weight,
        cost_compare.canonical_project_id,
    )
    delay_compare = _remaining_frame(cost_compare)
    prod_delay_pred = np.maximum(
        0.0, production_bundle["delay"].predict(delay_compare[delay_features])
    )
    remaining_pred = _aft_remaining_prediction(
        aft_models, delay_weights, delay_compare, delay_features
    )
    aft_delay_pred = _delay_from_remaining(delay_compare, remaining_pred)
    exp_delay_pred = np.maximum(
        0.0,
        aft_delay_pred
        + _corrections(delay_compare, aft_delay_pred, delay_calibration),
    )
    prod_delay = _regression_metrics(
        delay_compare.actual_delay_days,
        prod_delay_pred,
        delay_compare.sample_weight,
        delay_compare.canonical_project_id,
    )
    exp_delay = _regression_metrics(
        delay_compare.actual_delay_days,
        exp_delay_pred,
        delay_compare.sample_weight,
        delay_compare.canonical_project_id,
    )
    aft_before_calibration = _regression_metrics(
        delay_compare.actual_delay_days,
        aft_delay_pred,
        delay_compare.sample_weight,
        delay_compare.canonical_project_id,
    )

    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = (
        "PROMOTION CANDIDATE"
        if cost_gain >= 0
        and delay_gain >= 0
        and (cost_gain > 0 or delay_gain > 0)
        else "REGRESSION / DO NOT PROMOTE"
    )

    runtime_features = list(dict.fromkeys(cost_features + delay_features))
    lookup = {
        _key(row): {
            name: row.get(name)
            for name in runtime_features + ["snapshot_date", "planned_completion_date", "lifecycle_stage"]
        }
        for _, row in delay_compare.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": f"exp35-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment",
            "promotion_allowed": False,
            "baseline_contract": "current production = Exp12 Cost + Exp34 Delay",
            "components": {
                "cost": "Exp33 residual calibration only (Exp32 is Delay-only)",
                "delay": "Exp32 AFT remaining-time target followed by Exp33 residual calibration",
            },
            "delay_feature_contract": "current Exp34 production Delay features",
            "fixed_delay_blend_weights": delay_weights,
            "future_holdout_used_for_training_or_calibration": False,
            "cost_calibration": _public_calibration(cost_calibration),
            "delay_calibration": _public_calibration(delay_calibration),
            "cost_rolling_oof": cost_oof,
            "delay_aft_rolling_oof": delay_oof,
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "aft_delay_mae_before_exp33_calibration": aft_before_calibration["MAE"],
            "production_delay_full_shared_cohort_mae": shared_delay_full["MAE"],
            "comparison_test_projects": int(
                cost_compare.canonical_project_id.nunique()
            ),
            "comparison_test_snapshots": int(len(cost_compare)),
            "cost_comparison_projects": int(
                cost_compare.canonical_project_id.nunique()
            ),
            "cost_comparison_snapshots": int(len(cost_compare)),
            "delay_comparison_projects": int(
                delay_compare.canonical_project_id.nunique()
            ),
            "delay_comparison_snapshots": int(len(delay_compare)),
            "cost_comparison_cohort": "shared Exp12-comparable production cohort",
            "delay_comparison_cohort": "shared production cohort further filtered for AFT planned/remaining-time evidence",
            "decision": verdict,
        },
        "runtime_state": {
            "cost_model": production_bundle["cost"],
            "cost_features": cost_features,
            "cost_calibration": cost_calibration,
            "delay_models": aft_models,
            "delay_features": delay_features,
            "delay_weights": delay_weights,
            "delay_calibration": delay_calibration,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[
        frame.apply(lambda row: _key(row) in state["comparable"], axis=1)
    ].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError(
            "Exp35 Delay requires a snapshot with planned-completion evidence."
        )
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T

    prod_cost = float(
        state["cost_model"].predict(
            one.reindex(columns=state["cost_features"])
        )[0]
    )
    cost = prod_cost + float(
        _corrections(
            one,
            np.asarray([prod_cost], dtype=float),
            state["cost_calibration"],
        )[0]
    )

    remaining = _aft_remaining_prediction(
        state["delay_models"],
        state["delay_weights"],
        one,
        state["delay_features"],
    )
    aft_delay = _delay_from_remaining(one, remaining)
    delay = np.maximum(
        0.0,
        aft_delay
        + _corrections(one, aft_delay, state["delay_calibration"]),
    )[0]
    return {
        "predicted_cost_overrun": round(float(cost), 4),
        "predicted_delay_days": round(float(delay), 4),
        "predicted_remaining_days": round(float(remaining[0]), 4),
    }
