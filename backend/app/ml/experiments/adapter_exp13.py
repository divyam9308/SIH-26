"""Retrain & Compare adapter for Experiment 13."""
from __future__ import annotations

from backend.app.ml.experiments.trajectory_exp13 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    EXPERIMENT_SEQUENCE,
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production


def fit_against_production(**kwargs):
    # Production cost already uses the promoted Exp12 trajectory representation.
    # Exp13 must start from that exact evidence contract, then add only its new
    # regime/context features on top. This adapter remains experiment-only.
    enriched = dict(kwargs)
    enriched["data"] = enrich_supervised_for_production(kwargs["data"].copy())
    return fit_experiment(**enriched)
