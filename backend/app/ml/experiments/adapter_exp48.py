"""Discovery adapter for isolated Experiment 48."""
from backend.app.ml.experiments.spend_revision_leadlag_exp48 import EXPERIMENT_ID, EXPERIMENT_NAME, EXPERIMENT_SCOPE, EXPERIMENT_SEQUENCE, filter_comparable_rows, fit_experiment, predict_project


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
