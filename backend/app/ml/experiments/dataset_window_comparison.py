"""Evaluate the current production Cost/Delay stack across three fixed time windows.

This module intentionally contains no experiment-specific routing overrides. Each
window calls the same Exp105 Cost + Exp113 Delay production trainer used by main,
including the window-specific AFT routing policy introduced by PR #186.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.production_exp105_exp113_baseline import (
    PRODUCTION_COST_BASELINE,
    PRODUCTION_DELAY_BASELINE,
    train_window_with_promoted_cost_and_delay,
)

WINDOWS = {
    "2001_2017": {
        "training_start": 2001,
        "training_end": 2017,
        "test_start": 2018,
        "test_end": 2025,
    },
    "2001_2021": {
        "training_start": 2001,
        "training_end": 2021,
        "test_start": 2022,
        "test_end": 2025,
    },
    "2001_2022": {
        "training_start": 2001,
        "training_end": 2022,
        "test_start": 2023,
        "test_end": 2025,
    },
}


def get_window(label: str) -> dict:
    if label not in WINDOWS:
        raise ValueError(f"Unknown dataset window {label!r}; expected one of {sorted(WINDOWS)}")
    return dict(WINDOWS[label])


def _metric_value(metrics: dict, target: str, name: str) -> float:
    value = metrics[target][name]
    return float(value)


def run_window(
    label: str,
    *,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
) -> dict:
    """Run one fixed comparison window using the unmodified current production trainer."""
    window = get_window(label)
    if data is None or identity is None:
        built_data, built_identity = build_training_dataset()
        data = built_data if data is None else data
        identity = built_identity if identity is None else identity

    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix=f"dataset-window-{label}-") as tmp:
            return run_window(
                label,
                data=data,
                identity=identity,
                artifact_root=Path(tmp),
            )

    result = train_window_with_promoted_cost_and_delay(
        window["training_start"],
        window["training_end"],
        window["test_end"],
        data=data,
        identity=identity,
        artifact_root=artifact_root,
    )

    metrics = result["lifecycle"]["metrics"]
    metadata = result["metadata"]
    contract = dict(metadata.get("cost_evaluation_contract") or {})
    delay_contract = dict(metadata.get("delay_evaluation_contract") or {})

    return {
        "label": label,
        "training_window": f"{window['training_start']}-{window['training_end']}",
        "test_window": f"{window['test_start']}-{window['test_end']}",
        "training_start": window["training_start"],
        "training_end": window["training_end"],
        "test_start": window["test_start"],
        "test_end": window["test_end"],
        "cost_mae": _metric_value(metrics, "cost", "MAE"),
        "delay_mae_days": _metric_value(metrics, "delay", "MAE"),
        "cost_rmse": _metric_value(metrics, "cost", "RMSE"),
        "delay_rmse_days": _metric_value(metrics, "delay", "RMSE"),
        "comparison_projects": int(contract.get("test_projects", metrics["cost"].get("unique_projects", 0))),
        "comparison_snapshots": int(contract.get("test_snapshots", metrics["cost"].get("rows", 0))),
        "production_cost_baseline": metadata.get("production_cost_baseline", PRODUCTION_COST_BASELINE),
        "production_delay_baseline": metadata.get("production_delay_baseline", PRODUCTION_DELAY_BASELINE),
        "delay_routing_policy": delay_contract.get("routing_policy"),
        "delay_aft_projects": delay_contract.get("aft_eligible_projects"),
        "delay_fallback_projects": delay_contract.get("fallback_projects"),
        "production_main_contract": "PR186 window-specific AFT routing; no experiment-only patching",
    }


def write_result(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
