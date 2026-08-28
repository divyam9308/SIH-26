"""Retrain & Compare adapter for Experiment 29."""
from __future__ import annotations
from backend.app.ml.experiments.path_dependence_exp29 import EXPERIMENT_ID, EXPERIMENT_NAME, EXPERIMENT_SCOPE, EXPERIMENT_SEQUENCE, filter_comparable_rows, fit_experiment, predict_project

def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
