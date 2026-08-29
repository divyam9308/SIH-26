from backend.app.ml.experiments.reporting_behavior_cost_exp56 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,fit_experiment
def fit_against_production(**kwargs):return fit_experiment(**kwargs)
def filter_comparable_rows(frame,state):return frame.copy()
def predict_project(row,state):raise RuntimeError("Exp56 batch-evidence only until promotion")
