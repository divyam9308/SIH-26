"""Retrain & Compare adapter for Experiment 35."""
from backend.app.ml.experiments.cost_revision_hurdle_exp35 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    EXPERIMENT_SEQUENCE,
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
