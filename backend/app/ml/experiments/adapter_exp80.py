"""Discovery adapter for isolated Experiment 80."""
from backend.app.ml.experiments.exp80_revision_shock_cost import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    fit_experiment,
)

EXPERIMENT_SEQUENCE = 80
promotion_allowed = False


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)


def filter_comparable_rows(frame, state):
    return frame.copy()


def predict_project(row, state):
    raise RuntimeError("Experiment 80 is batch-evidence only until explicitly promoted.")
