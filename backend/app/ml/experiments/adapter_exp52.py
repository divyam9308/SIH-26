"""Discovery adapter for isolated Experiment 52."""
from backend.app.ml.experiments.delay_calibration_shrinkage_exp52 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,fit_experiment
def fit_against_production(**kwargs): return fit_experiment(**kwargs)
def filter_comparable_rows(frame,state): return frame.copy()
def predict_project(row,state): raise RuntimeError("Experiment 52 is batch-evidence only until explicitly promoted.")
