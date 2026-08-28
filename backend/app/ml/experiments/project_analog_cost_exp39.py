"""Experiment 39: historical project-analog Cost forecasting.

This is a non-parametric local forecaster. Production Cost features are
preprocessed using training data only, standardized, and queried against
historical training snapshots. Final Cost overrun is the distance-weighted
median outcome of the nearest historical analogs. Delay remains Exp34.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _preprocessor, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_39"
EXPERIMENT_NAME = "Historical project-analog Cost forecasting"
EXPERIMENT_SCOPE = "cost"
EXPERIMENT_SEQUENCE = 39
K_NEIGHBORS = 40


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cutoff = 0.5 * float(w.sum())
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(v[min(idx, len(v) - 1)])


def _analog_predict(preprocessor, scaler, nn, targets, project_weights, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = scaler.transform(preprocessor.transform(frame[features]))
    distances, indices = nn.kneighbors(matrix)
    output = []
    for dist, idx in zip(distances, indices):
        similarity = 1.0 / np.maximum(dist, 1e-6)
        weights = similarity * project_weights[idx]
        output.append(_weighted_median(targets[idx], weights))
    return np.asarray(output, dtype=float)


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    preprocessor = _preprocessor(train, cost_features)
    train_matrix = preprocessor.fit_transform(train[cost_features])
    scaler = StandardScaler(with_mean=False)
    train_matrix = scaler.fit_transform(train_matrix)
    k = min(K_NEIGHBORS, max(5, len(train)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(train_matrix)
    targets = train["actual_cost_overrun_percentage"].to_numpy(float)
    project_weights = train["sample_weight"].to_numpy(float)

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    exp_cost_pred = _analog_predict(preprocessor, scaler, nn, targets, project_weights, compare, cost_features)
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
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp39-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "local_nonparametric_project_analogs",
            "neighbors": k, "future_holdout_used_for_selection": False,
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
            "preprocessor": preprocessor, "scaler": scaler, "nn": nn,
            "targets": targets, "project_weights": project_weights, "cost_features": cost_features,
            "delay_models": delay_models, "delay_weights": delay_weights, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 39 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(_analog_predict(state["preprocessor"], state["scaler"], state["nn"], state["targets"], state["project_weights"], one, state["cost_features"])[0])
    delay = max(0.0, float(_blend_predict(state["delay_models"], state["delay_weights"], one, state["delay_features"])[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
