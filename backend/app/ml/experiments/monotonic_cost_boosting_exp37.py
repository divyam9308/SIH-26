"""Experiment 37: monotonic + interaction-disciplined Cost boosting.

Only the estimator geometry changes. The challenger uses the exact production
Cost feature contract, one-hot encoded from training categories only, and
XGBoost monotonic constraints for defensible as-of cost-stress signals. Delay
remains the promoted Exp34 production ensemble.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_37"
EXPERIMENT_NAME = "Monotonic constrained Cost boosting"
EXPERIMENT_SCOPE = "cost"
EXPERIMENT_SEQUENCE = 37
SEED = 26337
MONOTONIC_RAW = {
    "cost_escalation_percentage": 1,
    "expenditure_ratio": 1,
    "duration_ratio": 1,
    "schedule_slippage_days": 1,
}


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _fit_design(frame: pd.DataFrame, features: list[str]):
    raw = frame.reindex(columns=features).copy()
    categorical = [name for name in features if pd.api.types.is_object_dtype(raw[name]) or pd.api.types.is_string_dtype(raw[name])]
    for name in categorical:
        raw[name] = raw[name].astype("string").fillna("__MISSING__")
    for name in features:
        if name not in categorical:
            raw[name] = pd.to_numeric(raw[name], errors="coerce")
    design = pd.get_dummies(raw, columns=categorical, dummy_na=False, dtype=float)
    design = design.replace([np.inf, -np.inf], np.nan)
    medians = design.median(axis=0, numeric_only=True).fillna(0.0)
    design = design.fillna(medians).fillna(0.0)
    return design, categorical, medians


def _transform_design(frame: pd.DataFrame, features: list[str], categorical: list[str], columns: list[str], medians: pd.Series):
    raw = frame.reindex(columns=features).copy()
    for name in categorical:
        raw[name] = raw[name].astype("string").fillna("__MISSING__")
    for name in features:
        if name not in categorical:
            raw[name] = pd.to_numeric(raw[name], errors="coerce")
    design = pd.get_dummies(raw, columns=categorical, dummy_na=False, dtype=float)
    design = design.reindex(columns=columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    return design.fillna(medians).fillna(0.0)


def _constraints(columns: list[str]) -> tuple[int, ...]:
    values = []
    for name in columns:
        values.append(int(MONOTONIC_RAW.get(name, 0)))
    return tuple(values)


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    x_train, categorical, medians = _fit_design(train, cost_features)
    columns = list(x_train.columns)
    monotone = _constraints(columns)
    if not any(monotone):
        raise ValueError("Experiment 37 found no eligible monotonic Cost signals in the production contract.")
    model = XGBRegressor(
        n_estimators=320, learning_rate=0.03, max_depth=4, min_child_weight=4,
        subsample=0.9, colsample_bytree=0.9, objective="reg:absoluteerror",
        monotone_constraints=monotone, random_state=SEED, n_jobs=2,
    )
    model.fit(x_train, train["actual_cost_overrun_percentage"].to_numpy(float), sample_weight=train["sample_weight"].to_numpy(float))

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    x_compare = _transform_design(compare, cost_features, categorical, columns, medians)
    exp_cost_pred = model.predict(x_compare)
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
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp37-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "monotonic_constrained_cost_boosting",
            "monotonic_raw_features": {k: v for k, v in MONOTONIC_RAW.items() if k in cost_features},
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
            "cost_model": model, "cost_features": cost_features, "categorical": categorical,
            "columns": columns, "medians": medians,
            "delay_models": delay_models, "delay_weights": delay_weights, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 37 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    x = _transform_design(one, state["cost_features"], state["categorical"], state["columns"], state["medians"])
    cost = float(state["cost_model"].predict(x)[0])
    delay = max(0.0, float(_blend_predict(state["delay_models"], state["delay_weights"], one, state["delay_features"])[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
