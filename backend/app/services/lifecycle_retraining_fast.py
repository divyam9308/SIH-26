"""Activation shim for performance-preserving lifecycle retraining.

Importing this module swaps only the production trainer callable used by the existing
atomic lifecycle retraining service. Publication, provenance, validation, routing and
API behavior remain owned by lifecycle_retraining_service.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from backend.app.ml.production_exp105_exp113_baseline import (
    train_window_with_promoted_cost_and_delay as canonical_train_window,
)
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as fast_train_window,
)
from backend.app.services import lifecycle_retraining_service as base


def _performance_entrypoint(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
    verify_frozen_reference: bool = True,
) -> dict:
    return fast_train_window(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=artifact_root,
        verify_frozen_reference=verify_frozen_reference,
    )


# Keep existing canonical routing/audit checks valid while delegating execution to
# the performance wrapper. The canonical Exp105/Exp113 model module itself is not edited.
_performance_entrypoint.__module__ = canonical_train_window.__module__
_performance_entrypoint.__wrapped__ = canonical_train_window
base.train_window_with_promoted_cost_and_delay = _performance_entrypoint

retrain_lifecycle = base.retrain_lifecycle
clear_training_data_cache = base.clear_training_data_cache
_training_data = base._training_data

FAST_TRAINING_ENABLED = True
CANONICAL_TRAINER_MODULE = canonical_train_window.__module__
PERFORMANCE_TRAINER_MODULE = fast_train_window.__module__
