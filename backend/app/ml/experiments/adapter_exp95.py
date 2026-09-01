"""Discovery adapter for isolated Experiment 95."""
from backend.app.ml.experiments.exp95_dynamic_hierarchical_cost_priors import (
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
    raise RuntimeError("Experiment 95 is batch-evidence only until explicitly promoted.")
