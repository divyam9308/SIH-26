"""Retrain & Compare adapter for Experiment 7."""
from __future__ import annotations

from backend.app.ml.experiments.hierarchical_residual_priors_exp7 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    run_experiment,
)
from backend.app.ml.experiments.runtime_exp7 import (
    EXPERIMENT_SCOPE,
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)

EXPERIMENT_SEQUENCE = 7


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
