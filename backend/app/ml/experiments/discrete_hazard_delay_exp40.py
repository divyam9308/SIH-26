"""Experiment 40: discrete-time completion hazard model (Delay only).

Training snapshots are expanded into 90-day person-period rows. A classifier
estimates the conditional probability of completion in each future interval,
given survival to that interval. The hazard curve is converted to a median
remaining time and then to final Delay relative to the original planned
completion date. Cost remains production Exp12 unchanged.
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

EXPERIMENT_ID = "exp_40"
EXPERIMENT_NAME = "Discrete-time completion hazard model"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 40
SEED = 26440
INTERVAL_DAYS = 90
MAX_INTERVALS = 40


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _remaining_days(frame: pd.DataFrame) -> np.ndarray:
    completion = pd.to_datetime(frame["completion_date"], errors="coerce")
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    return (completion - snapshot).dt.days.to_numpy(float)


def expand_hazard_rows(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    chunks = []
    remaining = _remaining_days(train)
    for pos, (_, row) in enumerate(train.iterrows()):
        days = remaining[pos]
        if not np.isfinite(days) or days <= 0:
            continue
        event_interval = max(1, int(np.ceil(days / INTERVAL_DAYS)))
        periods = min(event_interval, MAX_INTERVALS)
        base = {name: row.get(name) for name in features}
        weight = float(row.sample_weight) / periods
        for interval in range(1, periods + 1):
            item = dict(base)
            item["exp40_interval_index"] = interval
            item["exp40_completion_event"] = int(event_interval <= MAX_INTERVALS and interval == event_interval)
            item["sample_weight"] = weight
            chunks.append(item)
    if not chunks:
        raise ValueError("Experiment 40 could not construct person-period rows.")
    return pd.DataFrame(chunks)


def _hazard_prediction(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    repeated = []
    for _, row in frame.iterrows():
        for interval in range(1, MAX_INTERVALS + 1):
            item = {name: row.get(name) for name in features}
            item["exp40_interval_index"] = interval
            repeated.append(item)
    expanded = pd.DataFrame(repeated)
    p = model.predict_proba(expanded[features + ["exp40_interval_index"]])[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6).reshape(len(frame), MAX_INTERVALS)
    survival = np.cumprod(1.0 - p, axis=1)
    cdf = 1.0 - survival
    med = []
    for curve in cdf:
        hit = np.flatnonzero(curve >= 0.5)
        interval = int(hit[0] + 1) if len(hit) else MAX_INTERVALS
        med.append(interval * INTERVAL_DAYS)
    return np.asarray(med, dtype=float)


def _final_delay_from_remaining(frame: pd.DataFrame, remaining: np.ndarray) -> np.ndarray:
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    planned = pd.to_datetime(frame["planned_completion_date"], errors="coerce")
    predicted_completion = snapshot + pd.to_timedelta(remaining, unit="D")
    return np.maximum(0.0, (predicted_completion - planned).dt.days.to_numpy(float))


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    expanded = expand_hazard_rows(train, delay_features)
    hazard_features = delay_features + ["exp40_interval_index"]
    classifier = LGBMClassifier(
        n_estimators=280, learning_rate=0.035, max_depth=5, num_leaves=24,
        class_weight="balanced", random_state=SEED, verbosity=-1,
    )
    hazard_model = _fit_pipeline(classifier, expanded, hazard_features, "exp40_completion_event")

    compare = _production_cost_evaluation_rows(test).copy()
    if compare["planned_completion_date"].isna().any():
        compare = compare[compare["planned_completion_date"].notna()].copy()
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    delay_weights, delay_oof = _oof_delay_weights(train, delay_features)
    delay_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(delay_models, delay_weights, compare, delay_features))
    exp_delay_pred = _final_delay_from_remaining(compare, _hazard_prediction(hazard_model, compare, delay_features))
    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_delay = _regression_metrics(compare["actual_delay_days"], exp_delay_pred, compare["sample_weight"], compare["canonical_project_id"])

    gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"
    lookup_features = list(dict.fromkeys(cost_features + delay_features + ["planned_completion_date"]))
    lookup = {_key(row): {name: row.get(name) for name in lookup_features} for _, row in compare.iterrows()}
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp40-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "discrete_time_completion_hazard",
            "interval_days": INTERVAL_DAYS, "max_intervals": MAX_INTERVALS,
            "person_period_rows": int(len(expanded)), "future_holdout_used_for_selection": False,
            "cost_policy": "production_exp12_retained_exactly", "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": prod_cost["MAE"],
            "cost_improvement_percentage": 0.0,
            "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(gain, 4),
            "comparison_test_projects": int(compare["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(compare)), "production_delay_blend_weights": delay_weights,
            "production_delay_rolling_oof": delay_oof, "decision": verdict,
        },
        "runtime_state": {
            "production_cost_model": production_bundle["cost"], "cost_features": cost_features,
            "hazard_model": hazard_model, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 40 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    remaining = _hazard_prediction(state["hazard_model"], one, state["delay_features"])
    delay = float(_final_delay_from_remaining(one, remaining)[0])
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
