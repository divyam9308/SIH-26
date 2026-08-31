"""Discovery adapter for isolated Experiment 86."""
from backend.app.ml.experiments.exp86_aft_disagreement_delay import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    fit_experiment,
)

EXPERIMENT_SEQUENCE = 86
promotion_allowed = False


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)


def filter_comparable_rows(frame, state):
    return frame.copy()


def predict_project(row, state):
    raise RuntimeError("Experiment 86 is batch-evidence only until explicitly promoted.")
