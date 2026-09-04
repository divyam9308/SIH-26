"""Compare optimized Exp105/Exp113 execution against the canonical trainer.

No MAE is hard-coded here. The canonical trainer on the current branch/data is the
reference for each requested window; the optimized path must reproduce its metrics
and persisted predictions within numerical tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.production_exp105_exp113_baseline import (
    train_window_with_promoted_cost_and_delay as canonical_train,
)
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as fast_train,
)

ATOL = 1e-6


def _metrics(result: dict) -> dict[str, float]:
    metrics = result["lifecycle"]["metrics"]
    return {
        "cost_mae": float(metrics["cost"]["MAE"]),
        "delay_mae": float(metrics["delay"]["MAE"]),
    }


def _assert_close(label: str, left: float, right: float) -> None:
    if not np.isclose(float(left), float(right), rtol=0.0, atol=ATOL):
        raise RuntimeError(f"{label} diverged: canonical={left} optimized={right}")


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "predicted_cost_overrun",
        "predicted_delay_days",
    ]
    return [col for col in candidates if col in frame.columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")

    with tempfile.TemporaryDirectory(prefix=f"canonical-{args.end}-") as canonical_td:
        canonical_started = time.perf_counter()
        canonical_result = canonical_train(
            args.start,
            args.end,
            args.test_end,
            data=data,
            identity=identity,
            artifact_root=Path(canonical_td),
        )
        canonical_seconds = time.perf_counter() - canonical_started
        canonical_validation = pd.read_csv(
            Path(canonical_td) / f"{args.start}_{args.end}" / "prediction_validation.csv"
        )

    with tempfile.TemporaryDirectory(prefix=f"optimized-{args.end}-") as optimized_td:
        optimized_started = time.perf_counter()
        optimized_result = fast_train(
            args.start,
            args.end,
            args.test_end,
            data=data,
            identity=identity,
            artifact_root=Path(optimized_td),
        )
        optimized_seconds = time.perf_counter() - optimized_started
        optimized_validation = pd.read_csv(
            Path(optimized_td) / f"{args.start}_{args.end}" / "prediction_validation.csv"
        )

    canonical_metrics = _metrics(canonical_result)
    optimized_metrics = _metrics(optimized_result)
    for name in canonical_metrics:
        _assert_close(name, canonical_metrics[name], optimized_metrics[name])

    if len(canonical_validation) != len(optimized_validation):
        raise RuntimeError(
            f"Prediction row count diverged: canonical={len(canonical_validation)} optimized={len(optimized_validation)}"
        )
    compared_columns = _prediction_columns(canonical_validation)
    if compared_columns != _prediction_columns(optimized_validation):
        raise RuntimeError("Canonical and optimized prediction schemas differ")
    max_prediction_delta = 0.0
    for col in compared_columns:
        left = pd.to_numeric(canonical_validation[col], errors="coerce").to_numpy(float)
        right = pd.to_numeric(optimized_validation[col], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, rtol=0.0, atol=ATOL, equal_nan=True):
            delta = np.nanmax(np.abs(left - right))
            raise RuntimeError(f"{col} diverged; max absolute delta={delta}")
        if len(left):
            delta = float(np.nanmax(np.abs(left - right))) if np.isfinite(left - right).any() else 0.0
            max_prediction_delta = max(max_prediction_delta, delta)

    payload = {
        "window": f"{args.start}_{args.end}",
        "test_end": args.test_end,
        "canonical": {**canonical_metrics, "elapsed_seconds": canonical_seconds},
        "optimized": {**optimized_metrics, "elapsed_seconds": optimized_seconds},
        "speedup": canonical_seconds / optimized_seconds if optimized_seconds > 0 else None,
        "prediction_columns_compared": compared_columns,
        "max_prediction_delta": max_prediction_delta,
        "tolerance": ATOL,
        "hard_coded_metric_gate": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print(f"CANONICAL_TOTAL_SECONDS={canonical_seconds:.3f}")
    print(f"FAST_CANONICAL_TOTAL_SECONDS={optimized_seconds:.3f}")
    print(f"FAST_CANONICAL_SPEEDUP={payload['speedup']:.3f}")
    print(f"CANONICAL_COST_MAE={canonical_metrics['cost_mae']}")
    print(f"FAST_COST_MAE={optimized_metrics['cost_mae']}")
    print(f"CANONICAL_DELAY_MAE={canonical_metrics['delay_mae']}")
    print(f"FAST_DELAY_MAE={optimized_metrics['delay_mae']}")


if __name__ == "__main__":
    main()
