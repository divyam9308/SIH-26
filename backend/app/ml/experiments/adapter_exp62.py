"""Discovery adapter for isolated Experiment 62."""
from backend.app.ml.experiments.u1_nonlinear_residual_exp62 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    EXPERIMENT_SEQUENCE,
    fit_experiment,
)


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)


def filter_comparable_rows(frame, state):
    return frame.copy()


def predict_project(row, state):
    raise RuntimeError("Experiment 62 is batch-evidence only until explicitly promoted.")
