"""Retrain & Compare adapter for Experiment 32."""
from backend.app.ml.experiments.aft_remaining_exp32 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,filter_comparable_rows,fit_experiment,predict_project
def fit_against_production(**kwargs): return fit_experiment(**kwargs)
