"""Discovery adapter for isolated Experiment 61."""
from backend.app.ml.experiments.combined_exp51_exp58_exp61 import EXPERIMENT_ID, EXPERIMENT_NAME, EXPERIMENT_SCOPE, EXPERIMENT_SEQUENCE, fit_experiment

def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)

def filter_comparable_rows(frame, state):
    return frame.copy()

def predict_project(row, state):
    raise RuntimeError("Experiment 61 is batch-evidence only until explicitly promoted.")
