"""Experiment 27: remaining schedule-error forecasting (Delay only).

Instead of relearning delay already acknowledged by the current official revised
completion date, predict the signed future schedule error:
    final actual delay - current as-of schedule slippage.
Final delay = current as-of slippage + predicted future schedule error.

Production Cost is retained unchanged. All feature values and the anchor are
available as-of the snapshot; future outcomes are used only as training labels.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_27"
EXPERIMENT_NAME = "Remaining schedule-error forecasting"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 27
DELAY_SEED = 26204


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    anchor = pd.to_numeric(frame.get("schedule_slippage_days"), errors="coerce")
    eligible = frame[anchor.notna()].copy()
    if eligible.empty:
        raise ValueError("Experiment 27 requires as-of schedule_slippage_days.")
    eligible["exp27_future_schedule_error"] = (
        pd.to_numeric(eligible.actual_delay_days, errors="coerce")
        - pd.to_numeric(eligible.schedule_slippage_days, errors="coerce")
    )
    eligible = eligible[eligible.exp27_future_schedule_error.notna()].copy()
    return assign_project_balanced_weights(eligible)


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched.snapshot_date, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    selected = dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    delay_name = selected.get("delay")
    if delay_name not in _regressors(DELAY_SEED):
        raise ValueError(f"Unsupported production Delay family for Exp27: {delay_name!r}")

    train_delay = _eligible(train)
    compare = _eligible(test)
    delay_features = list(contract["delay"])
    delay_model = _fit_pipeline(
        _regressors(DELAY_SEED)[delay_name],
        train_delay,
        delay_features,
        "exp27_future_schedule_error",
    )

    prod_delay_pred = np.maximum(0, production_bundle["delay"].predict(compare[delay_features]))
    residual_pred = delay_model.predict(compare[delay_features])
    exp_delay_pred = np.maximum(
        0,
        pd.to_numeric(compare.schedule_slippage_days, errors="coerce").to_numpy(dtype=float)
        + residual_pred,
    )
    prod_delay = _regression_metrics(
        compare.actual_delay_days, prod_delay_pred, compare.sample_weight, compare.canonical_project_id
    )
    exp_delay = _regression_metrics(
        compare.actual_delay_days, exp_delay_pred, compare.sample_weight, compare.canonical_project_id
    )

    cost_compare = _production_cost_evaluation_rows(test)
    cost_features = list(contract["cost"])
    prod_cost_pred = production_bundle["cost"].predict(cost_compare[cost_features])
    prod_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage,
        prod_cost_pred,
        cost_compare.sample_weight,
        cost_compare.canonical_project_id,
    )
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if delay_gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup = {
        _key(row): float(row.schedule_slippage_days)
        for _, row in compare.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": f"exp27-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment",
            "promotion_allowed": False,
            "changed_dimension": "delay_target_formulation",
            "hypothesis": "Predicting only schedule movement not already acknowledged in the current revised completion date reduces Delay MAE.",
            "selected_algorithms": selected,
            "delay_target": "actual_delay_days - schedule_slippage_days",
            "cost_policy": "production_retained",
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": prod_cost["MAE"],
            "cost_improvement_percentage": 0.0,
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(compare.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "training_delay_projects": int(train_delay.canonical_project_id.nunique()),
            "training_delay_snapshots": int(len(train_delay)),
            "delay_comparison_filter": "schedule_slippage_days available as-of snapshot",
            "cost_evaluation_cohort": "official production Exp12 comparable cohort",
            "decision": verdict,
        },
        "runtime_state": {
            "delay_model": delay_model,
            "delay_features": delay_features,
            "cost_model": production_bundle["cost"],
            "cost_features": cost_features,
            "anchors": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["anchors"]:
        raise ValueError("Experiment 27 requires a comparable snapshot with schedule slippage.")
    x_delay = row.to_frame().T.reindex(columns=state["delay_features"])
    residual = float(state["delay_model"].predict(x_delay)[0])
    delay = max(0.0, state["anchors"][key] + residual)
    x_cost = row.to_frame().T.reindex(columns=state["cost_features"])
    cost = float(state["cost_model"].predict(x_cost)[0])
    return {
        "predicted_cost_overrun": round(cost, 4),
        "predicted_delay_days": round(delay, 4),
        "predicted_future_schedule_error": round(residual, 4),
    }
