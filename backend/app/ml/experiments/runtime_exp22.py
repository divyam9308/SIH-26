"""Generic Retrain & Compare runtime for Experiment 22."""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.experiments.milestone_trajectory_exp22 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    MILESTONE_FEATURES,
    enrich_with_monthly_milestones,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED, enrich_supervised_for_production, target_feature_contract


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_with_monthly_milestones(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    metadata = production_bundle.get("metadata") or {}
    contract = target_feature_contract(metadata)
    selected = dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    if not selected.get("cost") or not selected.get("delay"):
        raise ValueError("Experiment 22 requires production-selected cost and delay algorithms.")

    cost_features = list(dict.fromkeys(contract["cost"] + MILESTONE_FEATURES))
    delay_features = list(dict.fromkeys(contract["delay"] + MILESTONE_FEATURES))
    cost_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[selected["cost"]], train, cost_features, "actual_cost_overrun_percentage")
    delay_model = _fit_pipeline(_regressors(26204)[selected["delay"]], train, delay_features, "actual_delay_days")

    prod_cost_pred = production_bundle["cost"].predict(test[contract["cost"]])
    prod_delay_pred = np.maximum(0, production_bundle["delay"].predict(test[contract["delay"]]))
    exp_cost_pred = cost_model.predict(test[cost_features])
    exp_delay_pred = np.maximum(0, delay_model.predict(test[delay_features]))
    prod_cost = _regression_metrics(test.actual_cost_overrun_percentage, prod_cost_pred, test.sample_weight, test.canonical_project_id)
    exp_cost = _regression_metrics(test.actual_cost_overrun_percentage, exp_cost_pred, test.sample_weight, test.canonical_project_id)
    prod_delay = _regression_metrics(test.actual_delay_days, prod_delay_pred, test.sample_weight, test.canonical_project_id)
    exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))

    lookup = {_key(row): {feature: row.get(feature) for feature in MILESTONE_FEATURES} for _, row in test.iterrows()}
    run_id = f"exp22-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "model_role": "experiment", "promotion_allowed": False,
            "added_features": MILESTONE_FEATURES, "selected_algorithms": selected,
            "metrics": {"cost": exp_cost, "delay": exp_delay},
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4), "improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(test.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(test)),
            "training_milestone_snapshot_share": float(train.exp22_milestone_ratio.notna().mean()),
            "test_milestone_snapshot_share": float(test.exp22_milestone_ratio.notna().mean()),
            "feature_history_granularity": "full official monthly history",
        },
        "runtime_state": {
            "cost_model": cost_model, "delay_model": delay_model,
            "cost_features": cost_features, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 22 milestone representation is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    cost_x = candidate.to_frame().T.reindex(columns=state["cost_features"])
    delay_x = candidate.to_frame().T.reindex(columns=state["delay_features"])
    return {
        "predicted_cost_overrun": round(float(state["cost_model"].predict(cost_x)[0]), 4),
        "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(delay_x)[0])), 4),
        "milestone_features_available": int(sum(pd.notna(candidate.get(feature)) for feature in MILESTONE_FEATURES)),
    }
