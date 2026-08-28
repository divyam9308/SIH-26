"""Experiment 44: DTW historical trajectory analogs (Delay only).

A fast nearest-neighbor shortlist is built from the production Delay feature
representation. Final analog ranking uses Dynamic Time Warping over each
project's past-only trajectory prefix for schedule, duration, progress,
expenditure and cost-escalation signals. Delay is the weighted median final
Delay of historical analogs. Cost remains production Exp12 unchanged.
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

EXPERIMENT_ID = "exp_44"
EXPERIMENT_NAME = "DTW historical trajectory analog Delay model"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 44
SIGNALS = [
    "schedule_slippage_days", "duration_ratio", "progress_deviation",
    "expenditure_ratio", "cost_escalation_percentage",
]
MAX_SEQUENCE = 8
SHORTLIST = 30


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    idx = int(np.searchsorted(np.cumsum(weights), 0.5 * float(weights.sum()), side="left"))
    return float(values[min(idx, len(values) - 1)])


def fit_signal_scaler(train: pd.DataFrame):
    medians, scales = {}, {}
    for name in SIGNALS:
        values = pd.to_numeric(train.get(name), errors="coerce")
        med = float(values.median()) if values.notna().any() else 0.0
        q1, q3 = values.quantile([0.25, 0.75]).tolist() if values.notna().any() else (0.0, 1.0)
        scale = float(q3 - q1)
        medians[name] = med
        scales[name] = scale if np.isfinite(scale) and scale > 1e-9 else 1.0
    return medians, scales


def build_sequences(frame: pd.DataFrame, medians: dict, scales: dict) -> dict[tuple[str, str], np.ndarray]:
    work = frame.copy()
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
    sequences: dict[tuple[str, str], np.ndarray] = {}
    for _, group in work.sort_values(["canonical_project_id", "snapshot_date"]).groupby("canonical_project_id", sort=False):
        history = []
        for _, row in group.iterrows():
            vector = []
            for name in SIGNALS:
                value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
                value = medians[name] if pd.isna(value) else float(value)
                vector.append((value - medians[name]) / scales[name])
            history.append(vector)
            sequences[_key(row)] = np.asarray(history[-MAX_SEQUENCE:], dtype=float)
    return sequences


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    band = max(2, abs(n - m) + 2)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = float(np.mean(np.abs(a[i - 1] - b[j - 1])))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def _analog_predictions(preprocessor, scaler, nn, train_sequences, train_targets, train_weights, query_sequences, frame, features):
    matrix = scaler.transform(preprocessor.transform(frame[features]))
    _, indices = nn.kneighbors(matrix)
    predictions = []
    for row_pos, (_, row) in enumerate(frame.iterrows()):
        query = query_sequences[_key(row)]
        idx = indices[row_pos]
        distances = np.asarray([dtw_distance(query, train_sequences[int(i)]) for i in idx], dtype=float)
        finite = np.isfinite(distances)
        if not finite.any():
            weights = train_weights[idx]
        else:
            weights = (1.0 / np.maximum(distances, 1e-4)) * train_weights[idx]
        predictions.append(_weighted_median(train_targets[idx], weights))
    return np.asarray(predictions, dtype=float)


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    medians, signal_scales = fit_signal_scaler(train)
    train_sequence_map = build_sequences(train, medians, signal_scales)
    test_sequence_map = build_sequences(test, medians, signal_scales)

    train_reset = train.reset_index(drop=True)
    train_sequences = [train_sequence_map[_key(row)] for _, row in train_reset.iterrows()]
    preprocessor = _preprocessor(train_reset, delay_features)
    train_matrix = preprocessor.fit_transform(train_reset[delay_features])
    scaler = StandardScaler(with_mean=False)
    train_matrix = scaler.fit_transform(train_matrix)
    k = min(SHORTLIST, max(5, len(train_reset)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(train_matrix)
    train_targets = train_reset["actual_delay_days"].to_numpy(float)
    train_weights = train_reset["sample_weight"].to_numpy(float)

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    prod_weights, prod_oof = _oof_delay_weights(train, delay_features)
    prod_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(prod_models, prod_weights, compare, delay_features))
    exp_delay_pred = np.maximum(0, _analog_predictions(
        preprocessor, scaler, nn, train_sequences, train_targets, train_weights,
        test_sequence_map, compare, delay_features,
    ))
    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_delay = _regression_metrics(compare["actual_delay_days"], exp_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup_features = list(dict.fromkeys(cost_features + delay_features))
    lookup = {
        _key(row): {"features": {name: row.get(name) for name in lookup_features}, "sequence": test_sequence_map[_key(row)]}
        for _, row in compare.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp44-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "dtw_past_trajectory_analog_delay",
            "signals": list(SIGNALS), "max_sequence_snapshots": MAX_SEQUENCE,
            "shortlist_size": k, "future_holdout_used_for_selection": False,
            "cost_policy": "production_exp12_retained_exactly", "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": prod_cost["MAE"],
            "cost_improvement_percentage": 0.0,
            "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(gain, 4),
            "comparison_test_projects": int(compare["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(compare)), "production_delay_blend_weights": prod_weights,
            "production_delay_rolling_oof": prod_oof, "decision": verdict,
        },
        "runtime_state": {
            "production_cost_model": production_bundle["cost"], "cost_features": cost_features,
            "preprocessor": preprocessor, "scaler": scaler, "nn": nn,
            "train_sequences": train_sequences, "train_targets": train_targets, "train_weights": train_weights,
            "delay_features": delay_features, "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 44 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key]["features"].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    query_sequences = {key: state["lookup"][key]["sequence"]}
    delay = float(_analog_predictions(
        state["preprocessor"], state["scaler"], state["nn"], state["train_sequences"],
        state["train_targets"], state["train_weights"], query_sequences, one, state["delay_features"],
    )[0])
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(max(0.0, delay), 4)}
