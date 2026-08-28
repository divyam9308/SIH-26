"""Retrain & Compare adapter for Experiment 33."""
from backend.app.ml.experiments.residual_calibration_exp33 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,filter_comparable_rows,fit_experiment,predict_project
def fit_against_production(**kwargs):return fit_experiment(**kwargs)
