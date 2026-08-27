"""Retrain & Compare adapter for Experiment 18 GRU-attention forecasting."""
from __future__ import annotations

EXPERIMENT_ID = "exp_18"
EXPERIMENT_NAME = "GRU attention monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 18


def fit_against_production(**kwargs):
    from backend.app.ml.experiments.gru_attention_exp18 import fit_experiment
    from backend.app.ml.production_cost_baseline import enrich_supervised_for_production
    enriched = dict(kwargs)
    enriched["data"] = enrich_supervised_for_production(kwargs["data"].copy())
    return fit_experiment(**enriched)


def filter_comparable_rows(frame, state):
    from backend.app.ml.experiments.gru_attention_exp18 import filter_comparable_rows as implementation
    return implementation(frame, state)


def predict_project(row, state):
    from backend.app.ml.experiments.gru_attention_exp18 import predict_project as implementation
    return implementation(row, state)
