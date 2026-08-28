"""Experiment 41: future schedule-extension hurdle model (Delay only).

Instead of directly regressing remaining schedule error (Exp27), this challenger
first estimates whether the project will incur another material extension and
then predicts the positive extension magnitude conditional on that event.
Final Delay = current as-of slippage + P(extension) * extension magnitude.
Cost remains production Exp12 unchanged; production Delay reference is Exp34.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_41"
EXPERIMENT_NAME = "Future schedule-extension hurdle model"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 41
SEED = 26441
EXTENSION_THRESHOLD_DAYS = 30.0


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def add_extension_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    current = pd.to_numeric(result.get("schedule_slippage_days"), errors="coerce")
    final = pd.to_numeric(result["actual_delay_days"], errors="coerce")
    result["exp41_current_slippage"] = current
    result["exp41_future_extension"] = final - current
    result["exp41_extension_event"] = result["exp41_future_extension"].gt(EXTENSION_THRESHOLD_DAYS).astype(int)
    return result


def _fit_classifier(train: pd.DataFrame, features: list[str]):
    labels = train["exp41_extension_event"]
    if labels.nunique() < 2:
        return None, float(labels.mean())
    classifier = RandomForestClassifier(
        n_estimators=320, min_samples_leaf=4, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=2,
    )
    return _fit_pipeline(classifier, train, features, "exp41_extension_event"), None


def _probability(model, constant, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if model is None:
        return np.full(len(frame), float(constant or 0.0))
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(frame))
    return model.predict_proba(frame[features])[:, classes.index(1)]


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    hurdle_train = add_extension_targets(train).dropna(subset=["exp41_current_slippage", "exp41_future_extension"])
    if hurdle_train["canonical_project_id"].nunique() < 10:
        raise ValueError("Experiment 41 has insufficient schedule-slippage history.")
    classifier, constant = _fit_classifier(hurdle_train, delay_features)
    event_train = hurdle_train[hurdle_train["exp41_extension_event"].eq(1)].copy()
    if event_train["canonical_project_id"].nunique() < 5:
        raise ValueError("Experiment 41 has insufficient material schedule-extension events.")
    magnitude_model = _fit_pipeline(_regressors(SEED)["lightgbm"], event_train, delay_features, "exp41_future_extension")
    lo, hi = np.nanpercentile(event_train["exp41_future_extension"].to_numpy(float), [1, 99])

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    prod_weights, prod_oof = _oof_delay_weights(train, delay_features)
    prod_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(prod_models, prod_weights, compare, delay_features))

    candidate_rows = add_extension_targets(compare)
    p = _probability(classifier, constant, candidate_rows, delay_features)
    magnitude = np.clip(magnitude_model.predict(candidate_rows[delay_features]), max(0.0, lo), max(0.0, hi))
    current = pd.to_numeric(candidate_rows["exp41_current_slippage"], errors="coerce").to_numpy(float)
    current = np.where(np.isfinite(current), current, prod_delay_pred)
    exp_delay_pred = np.maximum(0.0, current + p * magnitude)

    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_delay = _regression_metrics(compare["actual_delay_days"], exp_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup_features = list(dict.fromkeys(cost_features + delay_features))
    lookup = {
        _key(row): {"features": {name: row.get(name) for name in lookup_features}, "current": row.get("schedule_slippage_days")}
        for _, row in compare.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp41-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "future_schedule_extension_event_plus_magnitude",
            "extension_threshold_days": EXTENSION_THRESHOLD_DAYS,
            "training_extension_event_rate": float(hurdle_train["exp41_extension_event"].mean()),
            "future_holdout_used_for_selection": False, "cost_policy": "production_exp12_retained_exactly",
            "decision": verdict,
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
            "classifier": classifier, "constant": constant, "magnitude_model": magnitude_model,
            "magnitude_clip": (float(lo), float(hi)), "delay_features": delay_features,
            "production_delay_models": prod_models, "production_delay_weights": prod_weights,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 41 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key]["features"].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    prod_delay = max(0.0, float(_blend_predict(state["production_delay_models"], state["production_delay_weights"], one, state["delay_features"])[0]))
    p = float(_probability(state["classifier"], state["constant"], one, state["delay_features"])[0])
    magnitude = float(state["magnitude_model"].predict(one[state["delay_features"]])[0])
    lo, hi = state["magnitude_clip"]
    magnitude = float(np.clip(magnitude, max(0.0, lo), max(0.0, hi)))
    current = pd.to_numeric(pd.Series([state["lookup"][key]["current"]]), errors="coerce").iloc[0]
    anchor = float(current) if pd.notna(current) else prod_delay
    delay = max(0.0, anchor + p * magnitude)
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
