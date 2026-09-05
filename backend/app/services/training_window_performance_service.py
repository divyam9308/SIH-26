"""Persisted, like-for-like evaluation for the historical production windows."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from backend.app.core.config import MODELS_DIR


WINDOWS = ((2001, 2017), (2001, 2021), (2001, 2022))
_REQUIRED_COLUMNS = {
    "project_name", "completion_date", "actual_cost_overrun", "predicted_cost_overrun",
    "actual_delay_days", "predicted_delay_days",
}


def _artifact_path(window: str) -> Path:
    return MODELS_DIR / window / "prediction_validation.csv"


def _cohort_key(frame: pd.DataFrame) -> pd.Series:
    """Stable identity available in all three saved completed-project ledgers."""
    return frame["project_name"].astype(str).str.strip().str.upper() + "|" + frame["completion_date"].astype(str)


def training_window_performance() -> dict:
    frames: dict[str, pd.DataFrame] = {}
    timestamps: list[datetime] = []
    for start_year, end_year in WINDOWS:
        window = f"{start_year}_{end_year}"
        path = _artifact_path(window)
        if not path.exists():
            raise FileNotFoundError(f"Saved validation evidence for {window} is unavailable.")
        frame = pd.read_csv(path)
        missing = _REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Saved validation evidence for {window} is incomplete: {', '.join(sorted(missing))}.")
        completion = pd.to_datetime(frame["completion_date"], errors="coerce")
        frame = frame[completion.dt.year.between(2023, 2024)].copy()
        frame["_comparison_key"] = _cohort_key(frame)
        frames[window] = frame
        metadata_path = path.with_name("evaluation_results.json")
        if metadata_path.exists():
            created_at = (json.loads(metadata_path.read_text()).get("metadata") or {}).get("created_at")
            if created_at:
                timestamps.append(datetime.fromisoformat(created_at.replace("Z", "+00:00")))

    common_keys = set.intersection(*(set(frame["_comparison_key"]) for frame in frames.values()))
    if not common_keys:
        raise ValueError("No common completed-project cohort is available for the requested training windows.")

    results = []
    for start_year, end_year in WINDOWS:
        window = f"{start_year}_{end_year}"
        frame = frames[window].loc[frames[window]["_comparison_key"].isin(common_keys)].copy()
        results.append({
            "start_year": start_year,
            "end_year": end_year,
            "cost_mae": round(float(mean_absolute_error(frame.actual_cost_overrun, frame.predicted_cost_overrun)), 3),
            "delay_mae_days": round(float(mean_absolute_error(frame.actual_delay_days, frame.predicted_delay_days)), 3),
            "cost_r2": round(float(r2_score(frame.actual_cost_overrun, frame.predicted_cost_overrun)), 4),
            "delay_r2": round(float(r2_score(frame.actual_delay_days, frame.predicted_delay_days)), 4),
            "sample_count": int(len(frame)),
        })

    generated_at = max(timestamps).astimezone(timezone.utc).isoformat() if timestamps else datetime.fromtimestamp(
        max(_artifact_path(f"{start}_{end}").stat().st_mtime for start_, end in WINDOWS), timezone.utc
    ).isoformat()
    return {
        "windows": results,
        "evaluation_period": "2023-2024 completed projects",
        "sample_count": int(len(common_keys)),
        "generated_at": generated_at,
        "methodology": (
            "Saved canonical production prediction ledgers are filtered to the same completed-project cohort "
            "in 2023-2024. Cost MAE is percentage points; Delay MAE is days; R² is the cost-model R²."
        ),
    }
