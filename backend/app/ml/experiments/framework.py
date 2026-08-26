"""Shared contracts for isolated ML experiments.

Experiments must never overwrite production lifecycle artifacts.  This module
provides one neutral context/manifest contract so future hypotheses can be
compared on an explicitly frozen cohort and written below an immutable run_id.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.ml.monthly_training import MODEL_ROOT
from backend.app.ml.provenance import feature_schema_fingerprint, frame_fingerprint, git_commit_sha, new_run_id

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_MODEL_ROOT = MODEL_ROOT / "experiments"
ALLOWED_CHANGED_DIMENSIONS = {
    "algorithm",
    "cost_target",
    "delay_target",
    "feature_set",
    "loss_function",
    "model_routing",
    "sampling",
    "trajectory_regime_context",
    "weighting",
    "other",
}


@dataclass(frozen=True)
class ExperimentContext:
    experiment_id: str
    training_start: int
    training_end: int
    testing_start: int
    testing_end: int
    dataset_fingerprint: str
    training_fingerprint: str
    test_fingerprint: str
    feature_schema_fingerprint: str
    features: tuple[str, ...]
    weighting_policy: str
    baseline_name: str = "production_direct_final_overrun"

    @property
    def window(self) -> str:
        return f"{self.training_start}_{self.training_end}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = list(self.features)
        payload["window"] = self.window
        return payload


def build_experiment_context(
    *,
    experiment_id: str,
    full_data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    training_start: int,
    training_end: int,
    testing_end: int,
    weighting_policy: str,
    baseline_name: str = "production_direct_final_overrun",
) -> ExperimentContext:
    """Freeze the exact cohort/features an experiment is allowed to compare.

    This helper intentionally does not infer leakage from dataframe indices:
    callers may reset indices independently. Project-group/temporal leakage must
    be enforced by the split function that produced ``train`` and ``test``.
    """
    if int(training_start) > int(training_end):
        raise ValueError("training_start cannot be after training_end")
    if int(testing_end) <= int(training_end):
        raise ValueError("testing_end must be after training_end")
    if train.empty or test.empty:
        raise ValueError("experiment context requires non-empty train and test cohorts")
    return ExperimentContext(
        experiment_id=str(experiment_id),
        training_start=int(training_start),
        training_end=int(training_end),
        testing_start=int(training_end) + 1,
        testing_end=int(testing_end),
        dataset_fingerprint=frame_fingerprint(full_data),
        training_fingerprint=frame_fingerprint(train),
        test_fingerprint=frame_fingerprint(test),
        feature_schema_fingerprint=feature_schema_fingerprint(features),
        features=tuple(features),
        weighting_policy=str(weighting_policy),
        baseline_name=str(baseline_name),
    )


def experiment_run_directory(experiment_id: str, window: str, run_id: str) -> Path:
    """Return an immutable experiment-only artifact directory."""
    safe_id = str(experiment_id).strip().lower().replace(" ", "_")
    if not safe_id or "/" in safe_id or ".." in safe_id:
        raise ValueError("invalid experiment_id")
    safe_window = str(window)
    if "/" in safe_window or ".." in safe_window:
        raise ValueError("invalid experiment window")
    safe_run = str(run_id).strip()
    if not safe_run or "/" in safe_run or ".." in safe_run:
        raise ValueError("invalid run_id")
    return EXPERIMENT_MODEL_ROOT / safe_id / safe_window / safe_run


def new_experiment_manifest(
    *,
    context: ExperimentContext,
    name: str,
    changed_dimension: str,
    hypothesis: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create the standard metadata envelope for a candidate experiment run."""
    if changed_dimension not in ALLOWED_CHANGED_DIMENSIONS:
        raise ValueError(f"changed_dimension must be one of {sorted(ALLOWED_CHANGED_DIMENSIONS)}")
    run_id = run_id or new_run_id()
    return {
        "schema_version": 1,
        "model_role": "experiment",
        "experiment_id": context.experiment_id,
        "name": str(name),
        "run_id": run_id,
        "status": "COMPLETED",
        "decision": "PENDING",
        "promotion_allowed": False,
        "changed_dimension": changed_dimension,
        "hypothesis": str(hypothesis),
        "context": context.to_dict(),
        "source_commit": git_commit_sha(ROOT),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def assert_same_comparison_context(baseline: dict, candidate: dict) -> None:
    """Reject comparisons that do not share the frozen evidence contract."""
    required = (
        "training_fingerprint",
        "test_fingerprint",
        "feature_schema_fingerprint",
        "weighting_policy",
    )
    mismatches = [key for key in required if baseline.get(key) != candidate.get(key)]
    if mismatches:
        raise ValueError("experiment comparison context mismatch: " + ", ".join(mismatches))


def promotion_guard(manifest: dict) -> None:
    """Experiments cannot become production implicitly.

    A future explicit promotion workflow may call this only after a human marks
    the run ACCEPTED and sets promotion_allowed=true.  Merely training or
    persisting an experiment never satisfies the guard.
    """
    if manifest.get("model_role") != "experiment":
        raise ValueError("promotion guard expects an experiment manifest")
    if manifest.get("decision") != "ACCEPTED" or manifest.get("promotion_allowed") is not True:
        raise PermissionError("experiment is not explicitly accepted for promotion")
