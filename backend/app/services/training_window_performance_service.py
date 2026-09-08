"""Shared-holdout comparison used by the Training Window Performance page."""
from __future__ import annotations

from datetime import datetime, timezone


WINDOWS = (
    {"start_year": 2001, "end_year": 2017, "cost_mae": 42.160, "delay_mae_days": 494.779, "cost_r2": 0.5301},
    {"start_year": 2001, "end_year": 2021, "cost_mae": 24.266, "delay_mae_days": 343.592, "cost_r2": 0.7716},
    {"start_year": 2001, "end_year": 2022, "cost_mae": 23.750, "delay_mae_days": 294.287, "cost_r2": 0.8256},
)


def training_window_performance() -> dict:
    results = [
        {
            **window,
            "delay_r2": None,
            "sample_count": 0,
            "evaluation_period": "2023–2025",
            "source": "provided_shared_holdout_comparison",
        }
        for window in WINDOWS
    ]
    return {
        "windows": results,
        "evaluation_period": "All three training windows are evaluated on the shared 2023–2025 testing window.",
        "sample_count": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "All windows use the same 2023–2025 testing period. Cost MAE is shown as a percentage, "
            "Delay MAE is shown in days, and lower MAE indicates better predictive performance."
        ),
    }
