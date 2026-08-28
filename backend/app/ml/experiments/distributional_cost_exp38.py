"""Experiment 38: heteroscedastic distributional Cost regression.

A training-only forward fold estimates absolute Cost residual scale. A scale
model predicts conditional noise, then the final production-family Cost model
is refit with inverse-scale project weights, approximating the location term of
a Laplace location-scale likelihood. Delay remains current Exp34 production.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_38"
EXPERIMENT_NAME = "Heteroscedastic distributional Cost regression"
EXPERIMENT_SCOPE = "cost"
EXPERIMENT_SEQUENCE = 38
SEED = 26338


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _scale_weights(train: pd.DataFrame, features: list[str], family: str):
    year = int(pd.to_numeric(train["completion_year"], errors="coerce").max())
    fitting = train[pd.to_numeric(train["completion_year"], errors="coerce").lt(year)].copy()
    validation = train[pd.to_numeric(train["completion_year"], errors="coerce").eq(year)].copy()
    if fitting["canonical_project_id"].nunique() < 10 or validation["canonical_project_id"].nunique() < 3:
        raise ValueError("Experiment 38 requires a valid forward residual-scale fold.")
    pilot = _fit_pipeline(_regressors(SEED)[family], fitting, features, "actual_cost_overrun_percentage")
    residual = np.abs(validation["actual_cost_overrun_percentage"].to_numpy(float) - pilot.predict(validation[features]))
    validation["exp38_abs_residual_scale"] = np.maximum(1.0, residual)
    scale_model = _fit_pipeline(_regressors(SEED + 1)["extra_trees"], validation, features, "exp38_abs_residual_scale")
    predicted_scale = np.maximum(1.0, scale_model.predict(train[features]))
    low, high = np.nanpercentile(predicted_scale, [10, 90])
    clipped = np.clip(predicted_scale, max(1.0, low), max(1.0, high))
    inverse = 1.0 / clipped
    inverse = inverse / np.average(inverse, weights=train["sample_weight"].to_numpy(float))
    return scale_model, inverse, {"validation_year": year, "scale_p10": float(low), "scale_p90": float(high)}


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))
    family = str((production_bundle.get("metadata") or {}).get("selected_algorithms", {}).get("cost", "extra_trees"))

    scale_model, inverse_scale, scale_diag = _scale_weights(train, cost_features, family)
    weighted_train = train.copy()
    weighted_train["sample_weight"] = weighted_train["sample_weight"].to_numpy(float) * inverse_scale
    model = _fit_pipeline(_regressors(SEED + 2)[family], weighted_train, cost_features, "actual_cost_overrun_percentage")

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    exp_cost_pred = model.predict(compare[cost_features])
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], exp_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    delay_weights, delay_oof = _oof_delay_weights(train, delay_features)
    delay_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(delay_models, delay_weights, compare, delay_features))
    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])

    gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"
    lookup_features = list(dict.fromkeys(cost_features + delay_features))
    lookup = {_key(row): {name: row.get(name) for name in lookup_features} for _, row in compare.iterrows()}
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp38-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "heteroscedastic_inverse_scale_location_refit",
            "cost_family": family, "scale_diagnostics": scale_diag,
            "future_holdout_used_for_selection": False,
            "delay_policy": "current_exp34_production_retained_exactly", "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(gain, 4),
            "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": prod_delay["MAE"],
            "delay_improvement_percentage": 0.0,
            "comparison_test_projects": int(compare["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(compare)), "delay_blend_weights": delay_weights,
            "delay_rolling_oof": delay_oof, "decision": verdict,
        },
        "runtime_state": {
            "cost_model": model, "cost_features": cost_features, "scale_model": scale_model,
            "delay_models": delay_models, "delay_weights": delay_weights, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 38 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["cost_model"].predict(one[state["cost_features"]])[0])
    delay = max(0.0, float(_blend_predict(state["delay_models"], state["delay_weights"], one, state["delay_features"])[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
