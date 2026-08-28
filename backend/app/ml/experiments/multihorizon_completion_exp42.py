"""Experiment 42: multi-horizon completion classification (Delay only).

Independent training-only classifiers estimate P(completed within H days) for a
fixed set of future horizons. Their probabilities are forced non-decreasing in
H and converted to a median remaining time, then to final Delay relative to the
original planned completion date. Cost remains current production unchanged.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_42"
EXPERIMENT_NAME = "Multi-horizon completion probability model"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 42
SEED = 26442
HORIZONS = (180, 365, 548, 730, 1095, 1460, 2190)


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def remaining_days(frame: pd.DataFrame) -> np.ndarray:
    return (pd.to_datetime(frame["completion_date"], errors="coerce") - pd.to_datetime(frame["snapshot_date"], errors="coerce")).dt.days.to_numpy(float)


def monotonicize(probabilities: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.clip(probabilities, 0.0, 1.0), axis=1)


def _fit_horizon_model(train: pd.DataFrame, features: list[str], horizon: int):
    target = f"exp42_within_{horizon}d"
    labeled = train.copy()
    labeled[target] = (remaining_days(labeled) <= horizon).astype(int)
    if labeled[target].nunique() < 2:
        return None, float(labeled[target].mean())
    model = LGBMClassifier(
        n_estimators=240, learning_rate=0.04, max_depth=5, num_leaves=24,
        class_weight="balanced", random_state=SEED + horizon, verbosity=-1,
    )
    return _fit_pipeline(model, labeled, features, target), None


def _horizon_probabilities(models: dict, constants: dict, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    columns = []
    for horizon in HORIZONS:
        model = models[horizon]
        if model is None:
            p = np.full(len(frame), float(constants[horizon]))
        else:
            classes = list(model.named_steps["model"].classes_)
            p = model.predict_proba(frame[features])[:, classes.index(1)] if 1 in classes else np.zeros(len(frame))
        columns.append(p)
    return monotonicize(np.column_stack(columns))


def _median_remaining(probabilities: np.ndarray) -> np.ndarray:
    output = []
    for row in probabilities:
        hits = np.flatnonzero(row >= 0.5)
        if not len(hits):
            output.append(float(HORIZONS[-1]))
            continue
        idx = int(hits[0])
        if idx == 0:
            output.append(float(HORIZONS[0]))
            continue
        h0, h1 = HORIZONS[idx - 1], HORIZONS[idx]
        p0, p1 = row[idx - 1], row[idx]
        frac = 1.0 if p1 <= p0 else float(np.clip((0.5 - p0) / (p1 - p0), 0.0, 1.0))
        output.append(float(h0 + frac * (h1 - h0)))
    return np.asarray(output, dtype=float)


def _delay_from_remaining(frame: pd.DataFrame, days: np.ndarray) -> np.ndarray:
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    planned = pd.to_datetime(frame["planned_completion_date"], errors="coerce")
    completion = snapshot + pd.to_timedelta(days, unit="D")
    return np.maximum(0.0, (completion - planned).dt.days.to_numpy(float))


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    models, constants = {}, {}
    for horizon in HORIZONS:
        models[horizon], constants[horizon] = _fit_horizon_model(train, delay_features, horizon)

    compare = _production_cost_evaluation_rows(test).copy()
    compare = compare[compare["planned_completion_date"].notna()].copy()
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    prod_weights, prod_oof = _oof_delay_weights(train, delay_features)
    prod_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(prod_models, prod_weights, compare, delay_features))
    probabilities = _horizon_probabilities(models, constants, compare, delay_features)
    exp_delay_pred = _delay_from_remaining(compare, _median_remaining(probabilities))

    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_delay = _regression_metrics(compare["actual_delay_days"], exp_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup_features = list(dict.fromkeys(cost_features + delay_features + ["planned_completion_date"]))
    lookup = {_key(row): {name: row.get(name) for name in lookup_features} for _, row in compare.iterrows()}
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp42-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "multi_horizon_completion_classification",
            "horizons_days": list(HORIZONS), "future_holdout_used_for_selection": False,
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
            "models": models, "constants": constants, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 42 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    probs = _horizon_probabilities(state["models"], state["constants"], one, state["delay_features"])
    delay = float(_delay_from_remaining(one, _median_remaining(probs))[0])
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
