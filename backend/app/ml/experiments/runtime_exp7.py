"""Generic Retrain & Compare runtime for Experiment 7.

This module does not retrain production. The shared comparison harness first
freshly trains the current production stack, then passes that immutable bundle
here. Exp7 changes only the cost prediction by adding leakage-safe hierarchical
agency/sector residual priors learned from training projects.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.hierarchical_residual_priors_exp7 import (
    AGENCY_K,
    SECTOR_K,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    _corrections,
    _group_prior,
    _project_level_residuals,
)
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract

EXPERIMENT_SCOPE = "cost"


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _improvement(baseline: float, challenger: float) -> float:
    return (baseline - challenger) / baseline * 100.0 if baseline else 0.0


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(production_bundle.get("metadata") or {})

    cost_model = production_bundle["cost"]
    train_pred = cost_model.predict(train[contract["cost"]])
    prod_pred = cost_model.predict(test[contract["cost"]])

    project_residuals = _project_level_residuals(train, train_pred)
    agency_prior = _group_prior(project_residuals, "implementing_agency", AGENCY_K)
    sector_prior = _group_prior(project_residuals, "sector", SECTOR_K)
    correction = _corrections(test, agency_prior, sector_prior)
    exp_pred = prod_pred + correction

    prod_cost = _regression_metrics(
        test.actual_cost_overrun_percentage,
        prod_pred,
        test.sample_weight,
        test.canonical_project_id,
    )
    exp_cost = _regression_metrics(
        test.actual_cost_overrun_percentage,
        exp_pred,
        test.sample_weight,
        test.canonical_project_id,
    )
    gain = _improvement(float(prod_cost["MAE"]), float(exp_cost["MAE"]))

    production_delay_mae = None
    if "delay" in production_bundle:
        delay_pred = np.maximum(0, production_bundle["delay"].predict(test[contract["delay"]]))
        production_delay_mae = _regression_metrics(
            test.actual_delay_days,
            delay_pred,
            test.sample_weight,
            test.canonical_project_id,
        )["MAE"]

    predictions = {}
    for position, (_, row) in enumerate(test.iterrows()):
        predictions[_key(row)] = {
            "predicted_cost_overrun": float(exp_pred[position]),
            "hierarchical_cost_correction": float(correction[position]),
        }

    run_id = f"exp7-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    selected = dict(
        (production_bundle.get("metadata") or {}).get("selected_algorithms")
        or production_receipt.get("selected_algorithms")
        or {}
    )
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": run_id,
            "model_role": "experiment",
            "promotion_allowed": False,
            "selected_algorithms": selected,
            "metrics": {"cost": exp_cost},
            "delay_policy": "production_retained",
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(gain, 4),
            "improvement_percentage": round(gain, 4),
            "production_delay_mae": production_delay_mae,
            "delay_policy": "production_retained",
            "delay_experiment_status": "not_changed_by_exp7",
            "comparison_test_projects": int(test.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(test)),
            "training_projects_for_priors": int(project_residuals.canonical_project_id.nunique()),
            "agency_groups": len(agency_prior),
            "sector_groups": len(sector_prior),
            "nonzero_correction_share": float(np.mean(np.abs(correction) > 1e-12)),
            "mean_abs_correction": float(np.mean(np.abs(correction))),
        },
        "runtime_state": {
            "predictions": predictions,
            "comparable": set(predictions),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["predictions"]:
        raise ValueError("No Experiment 7 hierarchical correction is available for this snapshot.")
    result = dict(state["predictions"][key])
    result["predicted_cost_overrun"] = round(result["predicted_cost_overrun"], 4)
    result["delay_policy"] = "production_retained"
    return result
