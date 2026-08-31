"""Discovery adapter for isolated Experiment 82."""
from backend.app.ml.experiments.exp82_lifecycle_adaptive_cost_calibration import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    fit_experiment,
)

EXPERIMENT_SEQUENCE = 82
promotion_allowed = False


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)


def filter_comparable_rows(frame, state):
    return frame.copy()


def predict_project(row, state):
    raise RuntimeError("Experiment 82 is batch-evidence only until explicitly promoted.")
