"""Live retraining adapter for the official PAIMANA monthly lifecycle models.

Retraining is published atomically: a failed run never replaces the last known
complete model directory, and the visible training marker is always cleaned up.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from backend.app.ml.monthly_lifecycle import OUTCOMES, SNAPSHOTS, SNAPSHOTS_GZ, build_training_dataset
from backend.app.ml.monthly_training import MODEL_ROOT
from backend.app.ml.production_cost_baseline import target_feature_contract
from backend.app.ml.production_delay_baseline import train_window_with_promoted_cost_and_delay
from backend.app.ml.provenance import artifact_fingerprints, file_sha256
from backend.app.services import monthly_prediction_service

_REQUIRED_ARTIFACTS = [
    "cost_model.pkl",
    "delay_model.pkl",
    "risk_model.pkl",
    "metadata.json",
    "evaluation_results.json",
    "feature_quality_report.json",
    "shap_importance.json",
    "prediction_validation.csv",
]


def _source_dataset_files() -> dict[str, str | None]:
    snapshot_path = SNAPSHOTS if SNAPSHOTS.exists() else SNAPSHOTS_GZ
    return {
        "monthly_snapshots": file_sha256(snapshot_path) if snapshot_path.exists() else None,
        "completed_outcomes": file_sha256(OUTCOMES) if OUTCOMES.exists() else None,
    }


@lru_cache(maxsize=1)
def _cached_training_data() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    years = data["completion_year"].dropna().astype(int)
    if years.empty:
        raise ValueError("No identity-verified PAIMANA lifecycle outcomes are available for retraining.")
    return data, identity.copy(), int(years.min()), int(years.max())


def clear_training_data_cache() -> None:
    """Allow an explicit data refresh/rebuild process to invalidate this process cache."""
    _cached_training_data.cache_clear()


def _training_data() -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    data, identity, min_year, max_year = _cached_training_data()
    return data.copy(), identity.copy(), min_year, max_year


def _stamp_production_role(result: dict, target: Path) -> None:
    """Make the production/experiment boundary explicit in persisted artifacts."""
    metadata = result.get("metadata") or {}
    metadata["model_role"] = "production"
    result["metadata"] = metadata

    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        persisted_metadata = json.loads(metadata_path.read_text())
        persisted_metadata["model_role"] = "production"
        metadata_path.write_text(json.dumps(persisted_metadata, indent=2, allow_nan=False))

    evaluation_path = target / "evaluation_results.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        evaluation.setdefault("metadata", {})["model_role"] = "production"
        evaluation_path.write_text(json.dumps(evaluation, indent=2, allow_nan=False))


def _write_run_manifest(start_year: int, end_year: int, result: dict, target: Path | None = None) -> dict:
    target = target or (MODEL_ROOT / f"{start_year}_{end_year}")
    metadata = result.get("metadata") or {}
    lifecycle_metrics = (result.get("lifecycle") or {}).get("metrics") or {}
    feature_contract = target_feature_contract(metadata)
    missing = [name for name in _REQUIRED_ARTIFACTS if not (target / name).exists()]
    if missing:
        raise RuntimeError(f"Refusing to publish incomplete lifecycle run; missing artifacts: {', '.join(missing)}")
    provenance = dict(metadata.get("provenance") or {})
    run_id = metadata.get("run_id") or provenance.get("run_id")
    dataset_fingerprint = metadata.get("dataset_fingerprint") or provenance.get("dataset_fingerprint")
    if not run_id or not dataset_fingerprint:
        raise RuntimeError("Refusing to publish lifecycle run without run_id and dataset_fingerprint provenance.")
    payload = {
        "status": "complete",
        "model_role": "production",
        "model_family": "monthly_lifecycle",
        "model_version": metadata.get("model_version") or f"monthly-{start_year}-{end_year}",
        "run_id": run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "training_fingerprint": provenance.get("training_fingerprint"),
        "test_fingerprint": provenance.get("test_fingerprint"),
        "feature_schema_fingerprint": provenance.get("feature_schema_fingerprint"),
        "source_commit": provenance.get("source_commit"),
        "source_dataset_files": _source_dataset_files(),
        "window": f"{start_year}_{end_year}",
        "training_period": metadata.get("training_period") or [start_year, end_year],
        "testing_period": metadata.get("testing_period") or [],
        "feature_count": len(feature_contract["cost"]),
        "feature_count_by_target": {name: len(features) for name, features in feature_contract.items()},
        "production_cost_baseline": metadata.get("production_cost_baseline"),
        "production_delay_baseline": metadata.get("production_delay_baseline"),
        "promoted_from_experiment": metadata.get("promoted_from_experiment"),
        "promoted_delay_from_experiment": metadata.get("promoted_delay_from_experiment"),
        "metrics": {
            "cost_mae": (lifecycle_metrics.get("cost") or {}).get("MAE"),
            "delay_mae": (lifecycle_metrics.get("delay") or {}).get("MAE"),
            "risk_macro_f1": (lifecycle_metrics.get("risk") or {}).get("macro_f1"),
        },
        "artifacts": {name: True for name in _REQUIRED_ARTIFACTS},
        "artifact_fingerprints": artifact_fingerprints(target, _REQUIRED_ARTIFACTS),
        "created_at": metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }
    (target / "run_manifest.json").write_text(json.dumps(payload, indent=2))
    return payload


def _publish_staged_run(staged_target: Path, target: Path) -> None:
    """Replace a model directory only after a staged run is complete."""
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex[:10]}"
    had_previous = target.exists()
    if had_previous:
        target.rename(backup)
    try:
        staged_target.rename(target)
    except Exception:
        if had_previous and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def retrain_lifecycle(start_year: int, end_year: int) -> dict:
    """Retrain and atomically publish the monthly lifecycle model stack."""
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("Training start year must be less than or equal to training end year.")

    data, identity, min_year, max_year = _training_data()
    if end_year >= max_year:
        raise ValueError(f"Training must end before {max_year} so an unseen future lifecycle holdout remains.")
    if end_year < min_year or start_year > max_year:
        raise ValueError(f"Training range must overlap identity-verified lifecycle data ({min_year}-{max_year}).")

    selected_training_years = data.loc[data.completion_year.between(start_year, end_year), "completion_year"].dropna()
    if selected_training_years.empty:
        raise ValueError("The selected period has no identity-verified lifecycle training projects.")
    internal_validation_year = int(selected_training_years.max())

    window = f"{start_year}_{end_year}"
    target = MODEL_ROOT / window
    target.mkdir(parents=True, exist_ok=True)
    training_marker = target / ".training"
    training_marker.write_text(datetime.now(timezone.utc).isoformat())
    staging_root = MODEL_ROOT / ".staging" / f"{window}-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)

    try:
        result = train_window_with_promoted_cost_and_delay(
            start_year,
            end_year,
            max_year,
            data=data,
            identity=identity,
            artifact_root=staging_root,
        )
        staged_target = staging_root / window
        _stamp_production_role(result, staged_target)
        _write_run_manifest(start_year, end_year, result, staged_target)
        _publish_staged_run(staged_target, target)
    finally:
        training_marker.unlink(missing_ok=True)
        shutil.rmtree(staging_root, ignore_errors=True)

    metadata = result["metadata"]
    lifecycle = result["lifecycle"]
    lifecycle_metrics = lifecycle["metrics"]
    baseline_metrics = result["baseline"]["metrics"]
    feature_audit = metadata.get("feature_availability", {})
    selected = metadata.get("selected_algorithms", {})
    provenance = metadata.get("provenance", {})
    feature_contract = target_feature_contract(metadata)

    monthly_prediction_service._bundle.cache_clear()

    return {
        "status": "success",
        "model_role": "production",
        "model_family": "monthly_lifecycle",
        "model_version": metadata["model_version"],
        "run_id": metadata.get("run_id") or provenance.get("run_id"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint") or provenance.get("dataset_fingerprint"),
        "window": window,
        "training_years": f"{start_year}-{end_year}",
        "testing_years": f"{end_year + 1}-{max_year}",
        "training_samples": metadata["training_snapshots"],
        "training_projects": metadata["unique_training_projects"],
        "testing_samples": metadata["test_snapshots"],
        "testing_projects": metadata["unique_test_projects"],
        "features_used": metadata["features_used"],
        "cost_features_used": feature_contract["cost"],
        "delay_features_used": feature_contract["delay"],
        "risk_features_used": feature_contract["risk"],
        "feature_count": len(feature_contract["cost"]),
        "feature_count_by_target": {name: len(features) for name, features in feature_contract.items()},
        "production_cost_baseline": metadata.get("production_cost_baseline"),
        "production_delay_baseline": metadata.get("production_delay_baseline"),
        "promoted_from_experiment": metadata.get("promoted_from_experiment"),
        "promoted_delay_from_experiment": metadata.get("promoted_delay_from_experiment"),
        "selected_algorithms": {
            "cost": selected.get("cost"),
            "delay": selected.get("delay"),
            "risk": "random_forest",
        },
        "internal_validation_year": internal_validation_year,
        "future_holdout_start": end_year + 1,
        "future_holdout_end": max_year,
        "metrics": {
            "cost_model": lifecycle_metrics["cost"],
            "delay_model": lifecycle_metrics["delay"],
            "risk_model": lifecycle_metrics["risk"],
            "metadata": {
                "feature_count": len(feature_contract["cost"]),
                "feature_count_by_target": {name: len(features) for name, features in feature_contract.items()},
                "features_used": metadata["features_used"],
                "cost_features_used": feature_contract["cost"],
                "delay_features_used": feature_contract["delay"],
                "production_cost_baseline": metadata.get("production_cost_baseline"),
                "production_delay_baseline": metadata.get("production_delay_baseline"),
                "delay_evaluation_contract": metadata.get("delay_evaluation_contract"),
                "feature_quality": {
                    "data_quality_score": feature_audit.get("data_quality_score"),
                    "removed_invalid_feature_count": feature_audit.get("removed_invalid_feature_count", len(feature_audit.get("removed_features", []))),
                    "as_of_evidence_coverage": feature_audit.get("as_of_evidence_coverage"),
                },
                "leakage_policy": metadata.get("leakage_policy"),
                "snapshot_weighting_policy": metadata.get("snapshot_weighting_policy"),
                "balanced_stage_summary": metadata.get("balanced_stage_summary"),
            },
        },
        "baseline_comparison": {
            "feature_count": len(result["baseline"].get("features") or []),
            "cost_mae": baseline_metrics["cost"]["MAE"],
            "delay_mae": baseline_metrics["delay"]["MAE"],
            "risk_macro_f1": baseline_metrics["risk"]["macro_f1"],
            "purpose": "Controlled benchmark only; not the retrained production forecast model.",
        },
        "lifecycle_stages": lifecycle.get("lifecycle_stages", {}),
        "stage_distribution": lifecycle.get("stage_distribution", {}),
        "balanced_stage_summary": lifecycle.get("balanced_stage_summary", {}),
        "leakage_guard": "Future holdout projects are excluded from selection/fitting; direct features are same-snapshot, promoted cost trajectory features and Exp34 Delay path features use current/earlier snapshots, Delay blend weights are selected only on rolling folds inside the training period, and priors require prior completion.",
    }
