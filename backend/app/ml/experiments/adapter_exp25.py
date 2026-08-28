"""Retrain-and-compare adapter for Experiment 25."""
from __future__ import annotations

from backend.app.ml.experiments.milestone_delay_exp25 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
)
from backend.app.ml.experiments.runtime_exp25 import (
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)

EXPERIMENT_SEQUENCE = 25


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
