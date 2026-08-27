"""Retrain & Compare adapter for Experiment 19 small Transformer forecasting."""
from __future__ import annotations

EXPERIMENT_ID = "exp_19"
EXPERIMENT_NAME = "Small Transformer monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 19


def fit_against_production(**kwargs):
    from backend.app.ml.experiments.transformer_sequence_exp19 import fit_experiment
    from backend.app.ml.production_cost_baseline import enrich_supervised_for_production
    enriched = dict(kwargs)
    enriched["data"] = enrich_supervised_for_production(kwargs["data"].copy())
    return fit_experiment(**enriched)


def filter_comparable_rows(frame, state):
    from backend.app.ml.experiments.transformer_sequence_exp19 import filter_comparable_rows as implementation
    return implementation(frame, state)


def predict_project(row, state):
    from backend.app.ml.experiments.transformer_sequence_exp19 import predict_project as implementation
    return implementation(row, state)
