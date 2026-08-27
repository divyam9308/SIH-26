"""Generic Retrain & Compare runtime for Experiment 23."""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.experiments.geographic_priors_exp23 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    MAX_COST_CORRECTION,
    MAX_DELAY_CORRECTION,
    MISSING,
    STATE_K,
    STATE_SECTOR_K,
    _geo_key,
    _improvement,
    _prior,
    corrections,
    project_level_residuals,
)
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    if "state" not in enriched:
        enriched["state"] = None
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(production_bundle.get("metadata") or {})

    cost_model = production_bundle["cost"]
    delay_model = production_bundle["delay"]
    train_cost_pred = cost_model.predict(train[contract["cost"]])
    train_delay_pred = np.maximum(0, delay_model.predict(train[contract["delay"]]))
    prod_cost_pred = cost_model.predict(test[contract["cost"]])
    prod_delay_pred = np.maximum(0, delay_model.predict(test[contract["delay"]]))

    cost_projects = project_level_residuals(train, train_cost_pred, "actual_cost_overrun_percentage")
    delay_projects = project_level_residuals(train, train_delay_pred, "actual_delay_days")
    cost_state = _prior(cost_projects, "state", STATE_K, MAX_COST_CORRECTION)
    cost_ss = _prior(cost_projects, "state_sector", STATE_SECTOR_K, MAX_COST_CORRECTION)
    delay_state = _prior(delay_projects, "state", STATE_K, MAX_DELAY_CORRECTION)
    delay_ss = _prior(delay_projects, "state_sector", STATE_SECTOR_K, MAX_DELAY_CORRECTION)

    cost_corr = corrections(test, cost_state, cost_ss, MAX_COST_CORRECTION)
    delay_corr = corrections(test, delay_state, delay_ss, MAX_DELAY_CORRECTION)
    exp_cost_pred = prod_cost_pred + cost_corr
    exp_delay_pred = np.maximum(0, prod_delay_pred + delay_corr)

    prod_cost = _regression_metrics(test.actual_cost_overrun_percentage, prod_cost_pred, test.sample_weight, test.canonical_project_id)
    exp_cost = _regression_metrics(test.actual_cost_overrun_percentage, exp_cost_pred, test.sample_weight, test.canonical_project_id)
    prod_delay = _regression_metrics(test.actual_delay_days, prod_delay_pred, test.sample_weight, test.canonical_project_id)
    exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
    cost_gain = _improvement(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _improvement(float(prod_delay["MAE"]), float(exp_delay["MAE"]))

    predictions = {}
    for position, (_, row) in enumerate(test.iterrows()):
        predictions[_key(row)] = {
            "predicted_cost_overrun": float(exp_cost_pred[position]),
            "predicted_delay_days": float(exp_delay_pred[position]),
            "geographic_cost_correction": float(cost_corr[position]),
            "geographic_delay_correction": float(delay_corr[position]),
        }
    state_share = float(_geo_key(test["state"]).ne(MISSING).mean())
    run_id = f"exp23-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    selected = dict((production_bundle.get("metadata") or {}).get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": run_id,
            "model_role": "experiment",
            "promotion_allowed": False,
            "selected_algorithms": selected,
            "metrics": {"cost": exp_cost, "delay": exp_delay},
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(test.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(test)),
            "test_snapshot_state_share": state_share,
            "cost_nonzero_correction_share": float(np.mean(np.abs(cost_corr) > 1e-12)),
            "delay_nonzero_correction_share": float(np.mean(np.abs(delay_corr) > 1e-12)),
        },
        "runtime_state": {"predictions": predictions, "comparable": set(predictions)},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["predictions"]:
        raise ValueError("No Experiment 23 geographic correction is available for this snapshot.")
    result = dict(state["predictions"][key])
    result["predicted_cost_overrun"] = round(result["predicted_cost_overrun"], 4)
    result["predicted_delay_days"] = round(max(0.0, result["predicted_delay_days"]), 4)
    return result
