"""Experiment 36: leakage-safe ordered categorical Cost model.

Uses CatBoost's ordered categorical statistics on reusable project context
(agency/ministry/sector/state/lifecycle interactions) while excluding raw
project name and project identity. Delay remains current Exp34 production.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_36"
EXPERIMENT_NAME = "Ordered categorical context Cost model"
EXPERIMENT_SCOPE = "cost"
EXPERIMENT_SEQUENCE = 36
SEED = 26336
BASE_CONTEXT = ["sector", "ministry", "implementing_agency", "state", "lifecycle_stage"]
PAIR_CONTEXT = [
    ("implementing_agency", "sector", "exp36_agency_sector"),
    ("state", "sector", "exp36_state_sector"),
    ("ministry", "sector", "exp36_ministry_sector"),
]


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def enrich_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in BASE_CONTEXT:
        if name not in result:
            result[name] = "__MISSING__"
    for left, right, out in PAIR_CONTEXT:
        a = result[left].astype("string").fillna("__MISSING__")
        b = result[right].astype("string").fillna("__MISSING__")
        result[out] = a + "||" + b
    return result


def _prepare(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    out = frame.reindex(columns=features).copy()
    for name in categorical:
        if name in out:
            out[name] = out[name].astype("string").fillna("__MISSING__").astype(str)
    for name in features:
        if name not in categorical:
            out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_context(enrich_path_dependence(enrich_supervised_for_production(data.copy())))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    added = [name for name in BASE_CONTEXT if name in train] + [item[2] for item in PAIR_CONTEXT]
    features = list(dict.fromkeys(cost_features + added))
    categorical = [
        name for name in features
        if name in added or pd.api.types.is_object_dtype(train[name]) or pd.api.types.is_string_dtype(train[name])
    ]
    model = CatBoostRegressor(
        iterations=420, depth=6, learning_rate=0.035, loss_function="MAE",
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    x_train = _prepare(train, features, categorical)
    cat_idx = [features.index(name) for name in categorical]
    model.fit(
        x_train, train["actual_cost_overrun_percentage"].to_numpy(float),
        cat_features=cat_idx, sample_weight=train["sample_weight"].to_numpy(float),
    )

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    exp_cost_pred = model.predict(_prepare(compare, features, categorical))
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], exp_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    delay_weights, delay_oof = _oof_delay_weights(train, delay_features)
    delay_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(delay_models, delay_weights, compare, delay_features))
    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])

    gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"
    lookup_features = list(dict.fromkeys(features + delay_features))
    lookup = {_key(row): {name: row.get(name) for name in lookup_features} for _, row in compare.iterrows()}
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp36-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "ordered_categorical_context_model",
            "raw_project_name_used_as_feature": False, "categorical_features": categorical,
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
            "cost_model": model, "cost_features": features, "categorical": categorical,
            "delay_models": delay_models, "delay_weights": delay_weights, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 36 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["cost_model"].predict(_prepare(one, state["cost_features"], state["categorical"]))[0])
    delay = max(0.0, float(_blend_predict(state["delay_models"], state["delay_weights"], one, state["delay_features"])[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
