from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from backend.app.core.config import MODELS_DIR, PROCESSED_DIR
from backend.app.ml.real_time_windows import active_version


_WINDOW_VERSION = re.compile(r"^(?:monthly[-_])?(\d{4})[_-](\d{4})$")
_CANONICAL_LIFECYCLE_ONLY_WINDOWS = frozenset({"2001_2022"})


def _normalise_version(version: str | None) -> str | None:
    if not version:
        return None
    selected = version.strip()
    match = _WINDOW_VERSION.fullmatch(selected)
    return f"{match.group(1)}_{match.group(2)}" if match else selected


def _version(version: str | None = None) -> str | None:
    return _normalise_version(version or active_version())


def _model_path(version: str | None, *, explicit: bool) -> tuple[Path | None, str | None]:
    """Resolve lifecycle artifacts before legacy artifacts without cross-family fallback."""
    selected = _normalise_version(version)
    if not selected:
        return None, None
    lifecycle = MODELS_DIR / "monthly_lifecycle" / selected
    if lifecycle.is_dir() and any((lifecycle / name).exists() for name in ("evaluation_results.json", "prediction_validation.csv", "prediction_validation.csv.gz")):
        return lifecycle, "monthly_lifecycle"
    if selected in _CANONICAL_LIFECYCLE_ONLY_WINDOWS:
        raise ValueError(
            f"Canonical lifecycle evaluation for {selected} is unavailable; legacy completed-project artifacts are intentionally not selectable."
        )
    legacy = MODELS_DIR / selected
    if legacy.is_dir() and any((legacy / name).exists() for name in ("evaluation_results.json", "prediction_validation.csv", "evaluation_results.csv")):
        return legacy, "legacy"
    if explicit:
        raise FileNotFoundError(f"Requested model version {version} was not found.")
    return None, None


def _lifecycle_report(raw: dict, model_path: Path) -> dict:
    metadata = dict(raw.get("metadata") or {})
    lifecycle = raw.get("lifecycle") or {}
    metrics = lifecycle.get("metrics") or metadata.get("lifecycle_metrics") or {}
    feature_quality = dict(metadata.get("feature_availability") or {})
    quality_file = model_path / "feature_quality_report.json"
    if quality_file.exists():
        feature_quality.update(json.loads(quality_file.read_text()))
    features = list(metadata.get("features_used") or feature_quality.get("features_used") or [])
    training = metadata.get("training_period") or []
    testing = metadata.get("testing_period") or []
    manifest_path = model_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    metadata.update({
        "training_start": training[0] if len(training) > 0 else None,
        "training_end": training[1] if len(training) > 1 else None,
        "test_start": testing[0] if len(testing) > 0 else None,
        "test_end": testing[1] if len(testing) > 1 else None,
        "evaluated_test_start": testing[0] if len(testing) > 0 else None,
        "evaluated_test_end": testing[1] if len(testing) > 1 else None,
        "training_projects": metadata.get("unique_training_projects"),
        "evaluation_projects": metadata.get("unique_test_projects"),
        "feature_count": len(features),
        "feature_quality": {
            "data_quality_score": feature_quality.get("data_quality_score"),
            "removed_invalid_feature_count": feature_quality.get("removed_invalid_feature_count"),
            "as_of_evidence_coverage": feature_quality.get("as_of_evidence_coverage"),
        },
        "lifecycle_stage_metrics": lifecycle.get("lifecycle_stages") or metadata.get("lifecycle_stage_metrics") or {},
        "lifecycle_stage_distribution": lifecycle.get("stage_distribution") or metadata.get("lifecycle_stage_distribution") or {},
        "balanced_stage_summary": lifecycle.get("balanced_stage_summary") or metadata.get("balanced_stage_summary") or {},
        "provenance": {
            "run_id": metadata.get("run_id") or (metadata.get("provenance") or {}).get("run_id"),
            "dataset_fingerprint": metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint"),
            "manifest_status": manifest.get("status") if manifest else "legacy_missing_manifest",
        },
    })
    return {
        "model_family": "monthly_lifecycle",
        "model_version": metadata.get("model_version") or f"monthly-{raw.get('window', model_path.name).replace('_', '-')}",
        "metadata": metadata,
        "cost_model": metrics.get("cost", {}),
        "delay_model": metrics.get("delay", {}),
        "risk_model": metrics.get("risk", {}),
        "lifecycle_stages": metadata["lifecycle_stage_metrics"],
        "stage_distribution": metadata["lifecycle_stage_distribution"],
        "balanced_stage_summary": metadata["balanced_stage_summary"],
        "shap": raw.get("shap") or {},
        "sector_validation": None,
    }


def validation_report(version: str | None = None) -> dict:
    explicit = bool(version and version.strip())
    selected = _version(version)
    path, family = _model_path(selected, explicit=explicit)
    if path and family == "monthly_lifecycle":
        return _lifecycle_report(json.loads((path / "evaluation_results.json").read_text()), path)
    if path and family == "legacy":
        return json.loads((path / "evaluation_results.json").read_text())
    return json.loads((MODELS_DIR / "validation_report.json").read_text())


def _normalise_lifecycle_rows(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={
        "canonical_project_id": "project_id",
        "actual_cost_overrun_percentage": "actual_cost_overrun",
    })
    for column in ("predicted_cost_p10", "predicted_cost_p90", "predicted_delay_p10", "predicted_delay_p90", "model_confidence_percentage"):
        if column not in renamed:
            renamed[column] = None
    return renamed


def validation_rows(version: str | None = None) -> pd.DataFrame:
    explicit = bool(version and version.strip())
    selected = _version(version)
    path, family = _model_path(selected, explicit=explicit)
    if path:
        names = ("prediction_validation.csv", "prediction_validation.csv.gz", "evaluation_results.csv")
        for name in names:
            artifact = path / name
            if artifact.exists():
                frame = pd.read_csv(artifact, dtype={"project_id": str, "canonical_project_id": str})
                return _normalise_lifecycle_rows(frame) if family == "monthly_lifecycle" else frame
        if explicit:
            raise FileNotFoundError(f"Validation rows for requested model version {version} were not found.")
    return pd.read_csv(PROCESSED_DIR / "prediction_validation.csv", dtype={"project_id": str})


def validation_payload(limit: int = 100, version: str | None = None) -> dict:
    all_rows = validation_rows(version)
    frame = all_rows.head(max(1, min(limit, 500)))
    safe = frame.astype(object)
    safe = safe.where(~frame.isin([float("inf"), float("-inf")]), None)
    safe = safe.where(pd.notna(safe), None)
    return {"model_version": _version(version), "items": safe.to_dict(orient="records"), "total": int(len(all_rows))}


def rolling_validation_report(version: str | None = None) -> dict:
    explicit = bool(version and version.strip())
    selected = _version(version)
    path, _family = _model_path(selected, explicit=explicit)
    artifact = path / "rolling_validation_results.json" if path else None
    if not artifact or not artifact.exists():
        return {"model_version": selected, "folds": [], "fold_count": 0, "status": "not_generated"}
    return json.loads(artifact.read_text())
