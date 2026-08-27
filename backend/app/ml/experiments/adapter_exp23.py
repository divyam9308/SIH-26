"""Retrain-and-compare adapter for Experiment 23."""
from __future__ import annotations

from backend.app.ml.experiments.geographic_priors_exp23 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    run_experiment,
)
from backend.app.ml.experiments.runtime_exp23 import (
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)

EXPERIMENT_SEQUENCE = 23


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
