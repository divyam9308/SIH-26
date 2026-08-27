"""Retrain-and-compare adapter for Experiment 23."""
from __future__ import annotations

from backend.app.ml.experiments.geographic_priors_exp23 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    run_experiment,
)

EXPERIMENT_SEQUENCE = 23


def fit_against_production(*, training_start: int, training_end: int, test_end: int, **_: object) -> dict:
    return run_experiment(training_start, training_end, test_end)
