"""Discovery adapter for isolated Experiment 85."""
from backend.app.ml.experiments.exp85_revision_shock_delay import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    fit_experiment,
)

EXPERIMENT_SEQUENCE = 85
promotion_allowed = False


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)


def filter_comparable_rows(frame, state):
    return frame.copy()


def predict_project(row, state):
    raise RuntimeError("Experiment 85 is batch-evidence only until explicitly promoted.")
