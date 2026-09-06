"""Verified persisted lifecycle evaluations for selected training windows."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from backend.app.core.config import MODELS_DIR


WINDOWS = ((2001, 2020), (2001, 2021), (2001, 2022))


def _verified_artifact(window: str) -> Path:
    root = MODELS_DIR / "monthly_lifecycle" / window
    evaluation = root / "evaluation_results.json"
    manifest_path = root / "run_manifest.json"
    if not evaluation.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Verified lifecycle evaluation for {window} is unavailable.")

    manifest = json.loads(manifest_path.read_text())
    raw = json.loads(evaluation.read_text())
    metadata = raw.get("metadata") or {}
    if manifest.get("status") != "complete":
        raise ValueError(f"Lifecycle evaluation for {window} is not complete.")
    if not manifest.get("run_id") or manifest.get("run_id") != metadata.get("run_id"):
        raise ValueError(f"Lifecycle evaluation for {window} has invalid run provenance.")
    if not manifest.get("dataset_fingerprint") or manifest.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
        raise ValueError(f"Lifecycle evaluation for {window} has invalid dataset provenance.")
    return evaluation


def training_window_performance() -> dict:
    timestamps: list[datetime] = []
    results = []
    for start_year, end_year in WINDOWS:
        window = f"{start_year}_{end_year}"
        path = _verified_artifact(window)
        raw = json.loads(path.read_text())
        metadata = raw.get("metadata") or {}
        lifecycle = raw.get("lifecycle") or {}
        metrics = lifecycle.get("metrics") or {}
        cost = metrics.get("cost") or raw.get("cost_model") or {}
        delay = metrics.get("delay") or raw.get("delay_model") or {}
        created_at = metadata.get("created_at")
        if created_at:
            timestamps.append(datetime.fromisoformat(created_at.replace("Z", "+00:00")))
        results.append({
            "start_year": start_year,
            "end_year": end_year,
            "cost_mae": float(cost["MAE"]),
            "delay_mae_days": float(delay.get("MAE_days", delay["MAE"])),
            "cost_r2": float(cost["R2"]),
            "delay_r2": float(delay["R2"]),
            "sample_count": int(cost.get("unique_projects") or metadata.get("testing_samples") or 0),
            "evaluation_period": f"{(metadata.get('testing_period') or [metadata.get('evaluated_test_start'), metadata.get('evaluated_test_end')])[0]}–{(metadata.get('testing_period') or [metadata.get('evaluated_test_start'), metadata.get('evaluated_test_end')])[1]}",
            "source": "verified_canonical_monthly_lifecycle",
        })

    generated_at = max(timestamps).astimezone(timezone.utc).isoformat() if timestamps else datetime.fromtimestamp(
        max(_verified_artifact(f"{start}_{end}").stat().st_mtime for start, end in WINDOWS), timezone.utc
    ).isoformat()
    return {
        "windows": results,
        "evaluation_period": "Each artifact's own temporal holdout; cohorts differ by training window.",
        "sample_count": None,
        "generated_at": generated_at,
        "methodology": (
            "Only complete canonical monthly-lifecycle evaluations with matching run and dataset provenance are used. "
            "Each training window has its own future temporal holdout, so values are descriptive and not a shared-cohort ranking. "
            "Cost MAE is percentage points; Delay MAE is days; R² is the cost-model R²."
        ),
    }
