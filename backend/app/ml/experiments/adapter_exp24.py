"""Retrain-and-compare adapter for Experiment 24."""
from __future__ import annotations

from backend.app.ml.experiments.obstruction_reasons_exp24 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    filter_comparable_rows,
    fit_experiment,
    predict_project,
    run_experiment,
)

EXPERIMENT_SEQUENCE = 24


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
