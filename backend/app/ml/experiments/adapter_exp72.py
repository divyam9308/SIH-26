"""Discovery adapter for Experiment 72."""
from backend.app.ml.experiments.u11_u1_u3_delay_blend_exp72 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,fit_experiment

def fit_against_production(**kwargs):return fit_experiment(**kwargs)
def filter_comparable_rows(frame,state):return frame.copy()
def predict_project(row,state):raise RuntimeError('Experiment 72 is batch-evidence only until explicitly promoted.')
