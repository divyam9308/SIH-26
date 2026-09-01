"""Discovery adapter for isolated Experiment 100."""
from backend.app.ml.experiments.exp100_smooth_gam_cost import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    EXPERIMENT_SEQUENCE,
    fit_experiment,
)

PROMOTION_ALLOWED = False
promotion_allowed = False
scope = EXPERIMENT_SCOPE


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)


def filter_comparable_rows(frame, state):
    return frame.copy()


def predict_project(row, state):
    raise RuntimeError("Experiment 100 is batch-evidence only until explicitly promoted.")
