"""Discovery adapter for Experiment 73."""
from backend.app.ml.experiments.u12_u1_u10_cost_blend_exp73 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,fit_experiment

def fit_against_production(**kwargs):return fit_experiment(**kwargs)
def filter_comparable_rows(frame,state):return frame.copy()
def predict_project(row,state):raise RuntimeError('Experiment 73 is batch-evidence only until explicitly promoted.')
