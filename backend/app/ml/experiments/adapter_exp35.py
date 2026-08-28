"""Retrain-and-compare adapter for Experiment 35."""
from backend.app.ml.experiments.exp35_aft_residual_combo import (
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
