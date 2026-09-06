"""Latest persisted production evaluations for the requested training windows."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from backend.app.core.config import MODELS_DIR


WINDOWS = ((2001, 2017), (2001, 2021), (2001, 2022))
def _artifact_path(window: str) -> Path:
    return MODELS_DIR / window / "evaluation_results.json"


def _latest_artifact(window: str) -> tuple[Path, str]:
    lifecycle = MODELS_DIR / "monthly_lifecycle" / window / "evaluation_results.json"
    if lifecycle.exists():
        manifest = lifecycle.with_name("run_manifest.json")
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
            return lifecycle, "canonical_monthly_lifecycle"
    legacy = _artifact_path(window)
    if legacy.exists():
        return legacy, "persisted_production_evaluation"
    raise FileNotFoundError(f"Saved production evaluation for {window} is unavailable.")


def training_window_performance() -> dict:
    timestamps: list[datetime] = []
    results = []
    for start_year, end_year in WINDOWS:
        window = f"{start_year}_{end_year}"
        path, source = _latest_artifact(window)
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
            "source": source,
        })

    generated_at = max(timestamps).astimezone(timezone.utc).isoformat() if timestamps else datetime.fromtimestamp(
        max(_latest_artifact(f"{start}_{end}")[0].stat().st_mtime for start, end in WINDOWS), timezone.utc
    ).isoformat()
    return {
        "windows": results,
        "evaluation_period": "Each saved production artifact's latest temporal holdout",
        "sample_count": None,
        "generated_at": generated_at,
        "methodology": (
            "The newest complete canonical monthly-lifecycle evaluations are used where available; "
            "2001–2017 falls back to its newest persisted production evaluation because no equivalent lifecycle artifact exists. "
            "Cost MAE is percentage points; Delay MAE is days; R² is the cost-model R²."
        ),
    }
