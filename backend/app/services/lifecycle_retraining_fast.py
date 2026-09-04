"""Activation shim for performance-preserving lifecycle retraining.

Importing this module swaps only the production trainer callable used by the existing
atomic lifecycle retraining service. Publication, provenance, validation, routing and
API behavior remain owned by lifecycle_retraining_service.
"""
from __future__ import annotations

from backend.app.ml.production_exp105_exp113_baseline import (
    train_window_with_promoted_cost_and_delay as canonical_train_window,
)
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as fast_train_window,
)
from backend.app.services import lifecycle_retraining_service as base

# Keep the existing retrain_lifecycle function object and its atomic publish logic.
# Only replace the trainer global it resolves at execution time.
base.train_window_with_promoted_cost_and_delay = fast_train_window

retrain_lifecycle = base.retrain_lifecycle
clear_training_data_cache = base.clear_training_data_cache
_training_data = base._training_data

FAST_TRAINING_ENABLED = True
CANONICAL_TRAINER_MODULE = canonical_train_window.__module__
PERFORMANCE_TRAINER_MODULE = fast_train_window.__module__
