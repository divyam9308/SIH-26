from __future__ import annotations

import json
import re
from pathlib import Path

from backend.app.core.config import MODELS_DIR

_WINDOW = re.compile(r"^(\d{4})_(\d{4})$")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def lifecycle_runs(models_dir: Path | None = None) -> dict:
    """Discover lifecycle runs that actually exist on disk.

    The endpoint deliberately reflects runtime artifacts rather than a hard-coded
    list, so an arbitrary successful live retrain becomes selectable immediately.
    Older committed benchmark runs without model binaries/validation rows remain
    visible as summary-only runs when their evaluation report is available.
    """
    root = (models_dir or MODELS_DIR) / "monthly_lifecycle"
    if not root.exists():
        return {"items": [], "count": 0}

    items: list[dict] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = _WINDOW.fullmatch(path.name)
        if not match:
            continue

        start_year, end_year = (int(match.group(1)), int(match.group(2)))
        evaluation_path = path / "evaluation_results.json"
        metadata_path = path / "metadata.json"
        quality_path = path / "feature_quality_report.json"
        manifest_path = path / "run_manifest.json"
        validation_csv = path / "prediction_validation.csv"
        validation_gz = path / "prediction_validation.csv.gz"
        training_marker = path / ".training"

        evaluation = _read_json(evaluation_path)
        metadata = dict(evaluation.get("metadata") or _read_json(metadata_path))
        quality = dict(metadata.get("feature_availability") or {})
        quality.update(_read_json(quality_path))
        lifecycle_metrics = (evaluation.get("lifecycle") or {}).get("metrics") or metadata.get("lifecycle_metrics") or {}
        manifest = _read_json(manifest_path)

        has_evaluation = evaluation_path.exists()
        has_metadata = metadata_path.exists() or bool(metadata)
        has_validation_rows = validation_csv.exists() or validation_gz.exists()
        has_models = all((path / f"{name}_model.pkl").exists() for name in ("cost", "delay", "risk"))
        has_feature_quality = quality_path.exists() or bool(quality)
        has_shap = (path / "shap_importance.json").exists() or bool(evaluation.get("shap"))
        in_progress = training_marker.exists()
        manifest_complete = manifest.get("status") == "complete"
        complete = bool(
            not in_progress
            and has_evaluation
            and has_metadata
            and has_validation_rows
            and has_models
            and (manifest_complete or not manifest)
        )

        training = metadata.get("training_period") or [start_year, end_year]
        testing = metadata.get("testing_period") or []
        features = list(metadata.get("features_used") or quality.get("features_used") or [])
        cost = lifecycle_metrics.get("cost") or {}
        delay = lifecycle_metrics.get("delay") or {}
        risk = lifecycle_metrics.get("risk") or {}
        run_id = metadata.get("run_id") or manifest.get("run_id")
        dataset_fingerprint = metadata.get("dataset_fingerprint") or manifest.get("dataset_fingerprint")
        provenance_status = (
            "verified_runtime_run" if run_id and dataset_fingerprint
            else manifest.get("provenance_status") or "legacy_artifact_no_recorded_run_identity"
        )

        items.append({
            "window": path.name,
            "model_version": metadata.get("model_version") or f"monthly-{start_year}-{end_year}",
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "provenance_status": provenance_status,
            "training_start": training[0] if len(training) > 0 else start_year,
            "training_end": training[1] if len(training) > 1 else end_year,
            "testing_start": testing[0] if len(testing) > 0 else None,
            "testing_end": testing[1] if len(testing) > 1 else None,
            "feature_count": len(features),
            "data_quality_score": quality.get("data_quality_score"),
            "cost_mae": cost.get("MAE"),
            "delay_mae": delay.get("MAE"),
            "risk_macro_f1": risk.get("macro_f1"),
            "has_evaluation": has_evaluation,
            "has_validation_rows": has_validation_rows,
            "has_models": has_models,
            "has_feature_quality": has_feature_quality,
            "has_shap": has_shap,
            "has_manifest": manifest_path.exists(),
            "in_progress": in_progress,
            "complete": complete,
            "summary_available": has_evaluation,
            "status": "training" if in_progress else ("complete" if complete else ("summary_only" if has_evaluation else "incomplete")),
            # Metadata is the canonical source; a stale/test-written manifest must
            # never override the actual training artifact timestamp.
            "created_at": metadata.get("created_at") or manifest.get("created_at"),
        })

    items.sort(key=lambda item: (item["training_start"], item["training_end"]))
    return {"items": items, "count": len(items)}
