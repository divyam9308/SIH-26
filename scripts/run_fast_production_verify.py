"""Freshly run and self-verify the optimized Exp105/Exp113 execution path.

This verifier intentionally does not hard-code Cost or Delay MAE. The separate
canonical production-verification workflow validates the current canonical trainer
for the same windows. This job validates that the optimized execution path completes,
persists the same predictions it reports in-memory, preserves production metadata,
and records timing/parallelism information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights, build_training_dataset
from backend.app.ml.production_exp105_exp113_fast import train_window_with_promoted_cost_and_delay

ATOL = 1e-6


def _weighted_mae(frame: pd.DataFrame, prediction_col: str, actual_col: str) -> float:
    error = np.abs(
        pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(float)
        - pd.to_numeric(frame[actual_col], errors="coerce").to_numpy(float)
    )
    weights = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(float)
    return float(np.average(error, weights=weights))


def _assert_close(label: str, left: float, right: float) -> None:
    if not np.isclose(float(left), float(right), rtol=0.0, atol=ATOL):
        raise RuntimeError(f"{label} diverged: in_memory={left} persisted={right}")


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

    with tempfile.TemporaryDirectory(prefix=f"optimized-{args.end}-") as td:
        root = Path(td)
        started = time.perf_counter()
        result = train_window_with_promoted_cost_and_delay(
            args.start,
            args.end,
            args.test_end,
            data=data,
            identity=identity,
            artifact_root=root,
        )
        elapsed_seconds = time.perf_counter() - started

        target = root / f"{args.start}_{args.end}"
        validation = pd.read_csv(target / "prediction_validation.csv")
        comparable = validation[validation["cost_evaluation_eligible"].astype(bool)].copy()
        comparable = assign_project_balanced_weights(comparable)

        persisted_cost_mae = _weighted_mae(
            comparable, "predicted_cost_overrun", "actual_cost_overrun_percentage"
        )
        persisted_delay_mae = _weighted_mae(
            comparable, "predicted_delay_days", "actual_delay_days"
        )

    metrics = result["lifecycle"]["metrics"]
    cost_mae = float(metrics["cost"]["MAE"])
    delay_mae = float(metrics["delay"]["MAE"])
    _assert_close("Cost MAE", cost_mae, persisted_cost_mae)
    _assert_close("Delay MAE", delay_mae, persisted_delay_mae)

    promotion = result.get("promotion") or {}
    if promotion.get("risk_retained") is not True:
        raise RuntimeError("Optimized training failed Risk-isolation guard")

    metadata = result.get("metadata") or {}
    performance = result.get("performance") or metadata.get("training_performance") or {}
    if performance.get("model_logic") != "canonical_exp105_exp113_unchanged":
        raise RuntimeError("Optimized trainer did not report canonical model-logic preservation")
    if int(performance.get("fold_jobs", 0)) < 1 or int(performance.get("model_threads", 0)) < 1:
        raise RuntimeError("Optimized trainer did not report a valid worker layout")

    payload = {
        "window": f"{args.start}_{args.end}",
        "test_end": args.test_end,
        "cost_mae": cost_mae,
        "delay_mae": delay_mae,
        "persisted_cost_mae": persisted_cost_mae,
        "persisted_delay_mae": persisted_delay_mae,
        "elapsed_seconds": elapsed_seconds,
        "performance": performance,
        "production_cost_baseline": metadata.get("production_cost_baseline"),
        "production_delay_baseline": metadata.get("production_delay_baseline"),
        "hard_coded_metric_gate": False,
        "canonical_comparison_workflow": "Exp105 Cost + Exp113 Delay production verification",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print(f"FAST_CANONICAL_TOTAL_SECONDS={elapsed_seconds:.3f}")
    print(f"FAST_COST_MAE={cost_mae}")
    print(f"FAST_DELAY_MAE={delay_mae}")
    print(f"FAST_FOLD_JOBS={performance.get('fold_jobs')}")
    print(f"FAST_MODEL_THREADS={performance.get('model_threads')}")
    print(f"FAST_COST_OOF_CACHE_HIT={performance.get('cost_oof_cache_hit')}")
    print(f"FAST_DELAY_OOF_CACHE_HIT={performance.get('delay_oof_cache_hit')}")


if __name__ == "__main__":
    main()
