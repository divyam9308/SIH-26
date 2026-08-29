from backend.app.ml.experiments.pca_path_scores_exp59 import EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,EXPERIMENT_SEQUENCE,fit_experiment
def fit_against_production(**kwargs):return fit_experiment(**kwargs)
def filter_comparable_rows(frame,state):return frame.copy()
def predict_project(row,state):raise RuntimeError("Exp59 batch-evidence only until promotion")
