"""Retrain-and-compare adapter for Experiment 21."""
from __future__ import annotations

from backend.app.ml.experiments.scope_semantics_exp21 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    run_experiment,
)
from backend.app.ml.experiments.runtime_exp21 import (
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)

EXPERIMENT_SEQUENCE = 21


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
