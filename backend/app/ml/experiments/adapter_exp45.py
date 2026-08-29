"""Discovery adapter for isolated Experiment 45."""
from backend.app.ml.experiments.cost_revision_dynamics_exp45 import (
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
