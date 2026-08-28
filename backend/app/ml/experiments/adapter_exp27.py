"""Retrain & Compare adapter for Experiment 27."""
from __future__ import annotations

from backend.app.ml.experiments.remaining_delay_exp27 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    EXPERIMENT_SEQUENCE,
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
