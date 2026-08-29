"""Discovery adapter for isolated Experiment 50."""
from backend.app.ml.experiments.forward_schedule_revision_exp50 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,filter_comparable_rows,fit_experiment,predict_project
def fit_against_production(**kwargs):return fit_experiment(**kwargs)
