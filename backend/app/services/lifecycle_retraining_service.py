"""Live retraining adapter for the official PAIMANA monthly lifecycle models.

The website year-range selector must retrain the lifecycle cost, delay and risk
models, not the preserved five-feature completed-project baseline. This module
keeps that policy in one place, caches the prepared official lifecycle cohort for
repeated in-process retrains, and records successful arbitrary windows as real
runtime model runs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import uuid

import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import MODEL_ROOT, train_window
from backend.app.services import monthly_prediction_service


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
    # Train/evaluation code is allowed to mutate working frames, never the cache.
    return data.copy(), identity.copy(), min_year, max_year


def _dataset_fingerprint(data: pd.DataFrame) -> str:
    """Return a stable content fingerprint for the prepared lifecycle cohort.

    The hash deliberately covers every prepared column, not just row counts, so
    a changed PAIMANA value, identity link, target, feature or snapshot date
    changes the fingerprint. Column and row ordering are normalised first.
    """
    if data.empty:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    stable = data.copy()
    stable = stable.reindex(sorted(stable.columns), axis=1)
    sort_columns = [name for name in ("canonical_project_id", "snapshot_date", "completion_year") if name in stable]
    if sort_columns:
        stable = stable.sort_values(sort_columns, kind="mergesort", na_position="last")
    stable = stable.reset_index(drop=True)
    canonical = stable.astype("string").fillna("<NA>")
    row_hashes = pd.util.hash_pandas_object(canonical, index=False, categorize=True).to_numpy().tobytes()
    return "sha256:" + hashlib.sha256(row_hashes).hexdigest()


def _attach_run_provenance(start_year: int, end_year: int, result: dict, run_id: str, dataset_fingerprint: str) -> None:
    """Persist run identity into the canonical metadata/evaluation artifacts."""
    metadata = result.setdefault("metadata", {})
    metadata["run_id"] = run_id
    metadata["dataset_fingerprint"] = dataset_fingerprint
    result["run_id"] = run_id
    result["dataset_fingerprint"] = dataset_fingerprint
    target = MODEL_ROOT / f"{start_year}_{end_year}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (target / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))


def _write_run_manifest(start_year: int, end_year: int, result: dict) -> None:
    target = MODEL_ROOT / f"{start_year}_{end_year}"
    metadata = result.get("metadata") or {}
    lifecycle_metrics = (result.get("lifecycle") or {}).get("metrics") or {}
    artifacts = [
        "cost_model.pkl",
        "delay_model.pkl",
        "risk_model.pkl",
        "metadata.json",
        "evaluation_results.json",
        "feature_quality_report.json",
        "shap_importance.json",
        "prediction_validation.csv",
    ]
    run_id = metadata.get("run_id") or result.get("run_id")
    dataset_fingerprint = metadata.get("dataset_fingerprint") or result.get("dataset_fingerprint")
    payload = {
        "status": "complete",
        "model_family": "monthly_lifecycle",
        "model_version": metadata.get("model_version") or f"monthly-{start_year}-{end_year}",
        "window": f"{start_year}_{end_year}",
        "run_id": run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "provenance_status": "verified_runtime_run" if run_id and dataset_fingerprint else "legacy_artifact_no_recorded_run_identity",
        "training_period": metadata.get("training_period") or [start_year, end_year],
        "testing_period": metadata.get("testing_period") or [],
        "feature_count": len(metadata.get("features_used") or []),
        "metrics": {
            "cost_mae": (lifecycle_metrics.get("cost") or {}).get("MAE"),
            "delay_mae": (lifecycle_metrics.get("delay") or {}).get("MAE"),
            "risk_macro_f1": (lifecycle_metrics.get("risk") or {}).get("macro_f1"),
        },
        "artifacts": {name: (target / name).exists() for name in artifacts},
        "created_at": metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }
    (target / "run_manifest.json").write_text(json.dumps(payload, indent=2))


def retrain_lifecycle(start_year: int, end_year: int) -> dict:
    """Retrain the monthly lifecycle cost/delay/risk stack for a selected period.

    Algorithm selection remains internal-temporal: the latest completion year
    actually present inside the selected training range is used to choose the
    cost and delay regressor, then the winning regressors and the Random Forest
    risk classifier are fitted on the full selected training range. All later
    completion years remain future holdout data.
    """
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

    run_id = uuid.uuid4().hex
    dataset_fingerprint = _dataset_fingerprint(data)
    target = MODEL_ROOT / f"{start_year}_{end_year}"
    target.mkdir(parents=True, exist_ok=True)
    training_marker = target / ".training"
    training_marker.write_text(datetime.now(timezone.utc).isoformat())

    try:
        result = train_window(start_year, end_year, max_year, data=data, identity=identity)
        _attach_run_provenance(start_year, end_year, result, run_id, dataset_fingerprint)
        _write_run_manifest(start_year, end_year, result)
    finally:
        training_marker.unlink(missing_ok=True)

    metadata = result["metadata"]
    lifecycle = result["lifecycle"]
    lifecycle_metrics = lifecycle["metrics"]
    baseline_metrics = result["baseline"]["metrics"]
    feature_audit = metadata.get("feature_availability", {})
    selected = metadata.get("selected_algorithms", {})

    # Retraining can overwrite a previously loaded window; force inference to
    # reload the freshly written artifacts on the next forecast request.
    monthly_prediction_service._bundle.cache_clear()

    return {
        "status": "success",
        "model_family": "monthly_lifecycle",
        "model_version": metadata["model_version"],
        "window": f"{start_year}_{end_year}",
        "run_id": run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "created_at": metadata.get("created_at"),
        "training_years": f"{start_year}-{end_year}",
        "testing_years": f"{end_year + 1}-{max_year}",
        "training_samples": metadata["training_snapshots"],
        "training_projects": metadata["unique_training_projects"],
        "testing_samples": metadata["test_snapshots"],
        "testing_projects": metadata["unique_test_projects"],
        "features_used": metadata["features_used"],
        "feature_count": len(metadata["features_used"]),
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
                "feature_count": len(metadata["features_used"]),
                "features_used": metadata["features_used"],
                "feature_quality": {
                    "data_quality_score": feature_audit.get("data_quality_score"),
                    "removed_invalid_feature_count": feature_audit.get("removed_invalid_feature_count", len(feature_audit.get("removed_features", []))),
                },
                "run_id": run_id,
                "dataset_fingerprint": dataset_fingerprint,
                "created_at": metadata.get("created_at"),
                "leakage_policy": metadata.get("leakage_policy"),
                "snapshot_weighting_policy": metadata.get("snapshot_weighting_policy"),
            },
        },
        "baseline_comparison": {
            "feature_count": 5,
            "cost_mae": baseline_metrics["cost"]["MAE"],
            "delay_mae": baseline_metrics["delay"]["MAE"],
            "risk_macro_f1": baseline_metrics["risk"]["macro_f1"],
            "purpose": "Controlled benchmark only; not the retrained production forecast model.",
        },
        "lifecycle_stages": lifecycle.get("lifecycle_stages", {}),
        "leakage_guard": "The future holdout is excluded from algorithm selection and fitting; project identities may not cross the temporal split.",
    }
