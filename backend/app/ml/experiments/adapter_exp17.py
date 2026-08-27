"""Retrain & Compare adapter for Experiment 17 TCN sequence forecasting."""
from __future__ import annotations

EXPERIMENT_ID = "exp_17"
EXPERIMENT_NAME = "TCN monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 17


def fit_against_production(**kwargs):
    from backend.app.ml.experiments.tcn_sequence_exp17 import fit_experiment
    from backend.app.ml.production_cost_baseline import enrich_supervised_for_production
    enriched = dict(kwargs)
    enriched["data"] = enrich_supervised_for_production(kwargs["data"].copy())
    return fit_experiment(**enriched)


def filter_comparable_rows(frame, state):
    from backend.app.ml.experiments.tcn_sequence_exp17 import filter_comparable_rows as implementation
    return implementation(frame, state)


def predict_project(row, state):
    from backend.app.ml.experiments.tcn_sequence_exp17 import predict_project as implementation
    return implementation(row, state)
