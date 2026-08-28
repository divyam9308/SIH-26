"""Ablation: Exp33 residual calibration on the current Exp34 production Delay path.

Scientific contract
-------------------
* Current production is Exp12 Cost + Exp34 Delay.
* Cost is not challenged and must remain numerically identical.
* Delay predictions come from the unchanged Exp34 path-feature + OOF ensemble.
* The only challenger operation is Exp33's weighted-median residual correction.
* Calibration residuals are produced by rolling historical validation years;
  every family model is fitted on strictly earlier completion years.
* The current Exp34 production blend weights are held fixed so the residual
  calibration is learned against the same deployed Delay stack.
* The future holdout is never used to fit calibration.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    FAMILIES,
    PATH_FEATURES,
    _fit_delay_family_models,
    _rolling_folds,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_33_on_exp34"
EXPERIMENT_NAME = "Exp33 residual calibration on current Exp34 Delay"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 45
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
        raise ValueError("Exp33-on-Exp34 has insufficient finite OOF predictions.")

    prediction = pd.to_numeric(oof.loc[finite, "prediction"], errors="coerce")
    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, N_BINS + 1)).astype(float))
    if len(edges) < 3:
        edges = np.array([-np.inf, float(np.median(prediction)), np.inf])
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
        int(bin_id): _weighted_median(part["residual"], part["sample_weight"])
        for bin_id, part in work.groupby("bin")
    }
    stage_bin_medians: dict[tuple[str, int], float] = {}
    for (stage, bin_id), part in work.groupby(["lifecycle_stage", "bin"], dropna=False):
        if len(part) < MIN_GROUP_ROWS:
            continue
        stage_key = "<NA>" if pd.isna(stage) else str(stage)
        stage_bin_medians[(stage_key, int(bin_id))] = _weighted_median(
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


def _corrections(frame: pd.DataFrame, predictions: np.ndarray, calibration: dict) -> np.ndarray:
    edges = np.asarray(calibration["edges"], dtype=float)
    bins = np.digitize(np.asarray(predictions, dtype=float), edges[1:-1], right=False)
    result = np.zeros(len(frame), dtype=float)
    stages = frame.get("lifecycle_stage", pd.Series(pd.NA, index=frame.index))
    for index, (stage, bin_id) in enumerate(zip(stages, bins)):
        stage_key = "<NA>" if pd.isna(stage) else str(stage)
        correction = calibration["stage_bin_medians"].get((stage_key, int(bin_id)))
        if correction is None:
            correction = calibration["bin_medians"].get(
                int(bin_id), calibration["global_median"]
            )
        result[index] = float(correction)
    return result


def _exp34_delay_calibration_oof(
    train: pd.DataFrame,
    features: list[str],
    weights: dict[str, float],
) -> tuple[dict, list[dict]]:
    folds = _rolling_folds(train)
    if len(folds) < 2:
        raise ValueError("Exp33-on-Exp34 requires at least two rolling historical folds.")

    chunks: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for fitting, validation, year in folds:
        models = _fit_delay_family_models(fitting, features)
        prediction = np.zeros(len(validation), dtype=float)
        for family in FAMILIES:
            prediction += float(weights[family]) * models[family].predict(validation[features])
        prediction = np.maximum(0.0, prediction)

        chunk = validation[
            ["actual_delay_days", "sample_weight", "canonical_project_id", "lifecycle_stage"]
        ].copy()
        chunk["prediction"] = prediction
        chunk["residual"] = (
            pd.to_numeric(chunk["actual_delay_days"], errors="coerce") - prediction
        )
        chunks.append(chunk)
        diagnostics.append(
            {
                "year": int(year),
                "projects": int(validation["canonical_project_id"].nunique()),
                "MAE_before_calibration": _regression_metrics(
                    validation["actual_delay_days"],
                    prediction,
                    validation["sample_weight"],
                    validation["canonical_project_id"],
                )["MAE"],
            }
        )

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
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))
    delay_weights = {
        family: float((metadata.get("delay_blend_weights") or {}).get(family, 0.0))
        for family in FAMILIES
    }
    if abs(sum(delay_weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            "Exp33-on-Exp34 requires normalized current production Exp34 blend weights; "
            f"got {delay_weights}."
        )

    delay_calibration, delay_oof = _exp34_delay_calibration_oof(
        train, delay_features, delay_weights
    )

    # Use the exact verified shared Exp12/Exp34 production cohort for both targets.
    compare = _production_cost_evaluation_rows(test)

    production_cost_prediction = production_bundle["cost"].predict(compare[cost_features])
    experiment_cost_prediction = production_cost_prediction.copy()
    production_cost = _regression_metrics(
        compare["actual_cost_overrun_percentage"],
        production_cost_prediction,
        compare["sample_weight"],
        compare["canonical_project_id"],
    )
    experiment_cost = _regression_metrics(
        compare["actual_cost_overrun_percentage"],
        experiment_cost_prediction,
        compare["sample_weight"],
        compare["canonical_project_id"],
    )

    production_delay_prediction = np.maximum(
        0.0, production_bundle["delay"].predict(compare[delay_features])
    )
    delay_correction = _corrections(compare, production_delay_prediction, delay_calibration)
    experiment_delay_prediction = np.maximum(
        0.0, production_delay_prediction + delay_correction
    )
    production_delay = _regression_metrics(
        compare["actual_delay_days"],
        production_delay_prediction,
        compare["sample_weight"],
        compare["canonical_project_id"],
    )
    experiment_delay = _regression_metrics(
        compare["actual_delay_days"],
        experiment_delay_prediction,
        compare["sample_weight"],
        compare["canonical_project_id"],
    )

    cost_gain = _gain(float(production_cost["MAE"]), float(experiment_cost["MAE"]))
    delay_gain = _gain(float(production_delay["MAE"]), float(experiment_delay["MAE"]))
    if abs(cost_gain) > 1e-12:
        raise AssertionError(
            f"Delay-only Exp33-on-Exp34 changed Cost MAE unexpectedly: {cost_gain}"
        )
    verdict = "PROMOTION CANDIDATE" if delay_gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup_features = list(dict.fromkeys(cost_features + delay_features + ["lifecycle_stage"]))
    lookup = {
        _key(row): {name: row.get(name) for name in lookup_features}
        for _, row in compare.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": f"exp33-on-exp34-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment",
            "promotion_allowed": False,
            "baseline_contract": "current production = Exp12 Cost + Exp34 Delay",
            "changed_dimension": "exp33_cross_fitted_post_model_delay_calibration_only",
            "cost_policy": "production_exp12_retained_exactly",
            "delay_policy": "current_exp34_prediction_plus_exp33_weighted_median_residual",
            "fixed_delay_blend_weights": delay_weights,
            "delay_calibration": _public_calibration(delay_calibration),
            "delay_rolling_oof": delay_oof,
            "future_holdout_used_for_calibration": False,
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": production_cost["MAE"],
            "experiment_cost_mae": experiment_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": production_delay["MAE"],
            "experiment_delay_mae": experiment_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(compare["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "comparison_cohort": "exact shared Exp12/Exp34 production evaluation cohort",
            "decision": verdict,
        },
        "runtime_state": {
            "cost_model": production_bundle["cost"],
            "delay_model": production_bundle["delay"],
            "cost_features": cost_features,
            "delay_features": delay_features,
            "delay_calibration": delay_calibration,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Exp33-on-Exp34 feature vector is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T

    production_cost = float(
        state["cost_model"].predict(one.reindex(columns=state["cost_features"]))[0]
    )
    production_delay = max(
        0.0,
        float(state["delay_model"].predict(one.reindex(columns=state["delay_features"]))[0]),
    )
    correction = float(
        _corrections(
            one,
            np.asarray([production_delay], dtype=float),
            state["delay_calibration"],
        )[0]
    )
    return {
        "predicted_cost_overrun": round(production_cost, 4),
        "predicted_delay_days": round(max(0.0, production_delay + correction), 4),
        "delay_median_residual_correction": round(correction, 4),
    }
