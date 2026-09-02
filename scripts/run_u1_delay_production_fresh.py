"""Freshly train and verify the promoted Exp105 Cost + Exp113 Delay production stack."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights, build_training_dataset
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay

EXPECTED_COST = {
    2019: 27.801,
    2021: 25.829,
}
TOLERANCE = 0.05


def _close(actual: float, expected: float) -> bool:
    return abs(float(actual) - float(expected)) <= TOLERANCE


def _weighted_mae(frame: pd.DataFrame, prediction_col: str, actual_col: str) -> float:
    error = np.abs(
        pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(float)
        - pd.to_numeric(frame[actual_col], errors="coerce").to_numpy(float)
    )
    return float(np.average(error, weights=frame["sample_weight"].to_numpy(float)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--test-end", type=int, default=2025)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.start != 2001 or a.end not in EXPECTED_COST or a.test_end != 2025:
        raise ValueError("Production verification supports only 2001-2019 and 2001-2021 through 2025")

    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    with tempfile.TemporaryDirectory(prefix=f"exp105-exp113-production-{a.end}-") as td:
        artifact_root = Path(td)
        result = train_window_with_promoted_cost_and_delay(
            a.start, a.end, a.test_end, data=data, identity=identity, artifact_root=artifact_root
        )
        validation = pd.read_csv(artifact_root / f"{a.start}_{a.end}" / "prediction_validation.csv")
        comparable = validation[validation["cost_evaluation_eligible"].astype(bool)].copy()
        comparable = assign_project_balanced_weights(comparable)
        live_cost_mae = _weighted_mae(
            comparable, "predicted_cost_overrun_percentage", "actual_cost_overrun_percentage"
        )
        live_delay_mae = _weighted_mae(comparable, "predicted_delay_days", "actual_delay_days")

    metrics = result["lifecycle"]["metrics"]
    promo = result["promotion"]
    contract = result["metadata"]["cost_evaluation_contract"]
    payload = {
        "window": f"{a.start}_{a.end}",
        "test_end": a.test_end,
        "previous_cost_mae": promo["cost"]["previous_cost_mae"],
        "cost_mae": metrics["cost"]["MAE"],
        "persisted_inference_cost_mae": live_cost_mae,
        "cost_improvement_percentage": promo["cost"]["cost_improvement_percentage"],
        "previous_delay_mae": promo["delay"]["previous_delay_mae"],
        "delay_mae": metrics["delay"]["MAE"],
        "delay_mape_percent": metrics["delay"].get("MAPE"),
        "persisted_inference_delay_mae": live_delay_mae,
        "delay_improvement_percentage": promo["delay"]["delay_improvement_percentage"],
        "delay_routing_contract": result["metadata"].get("delay_evaluation_contract"),
        "comparison_test_projects": contract["test_projects"],
        "comparison_test_snapshots": contract["test_snapshots"],
        "production_cost_baseline": result["metadata"]["production_cost_baseline"],
        "production_delay_baseline": result["metadata"]["production_delay_baseline"],
        "promoted_cost_from_experiment": result["metadata"]["promoted_cost_from_experiment"],
        "promoted_delay_from_experiment": result["metadata"]["promoted_delay_from_experiment"],
        "risk_retained": promo["risk_retained"],
    }

    expected_cost = EXPECTED_COST[a.end]
    if not _close(payload["cost_mae"], expected_cost):
        raise RuntimeError(f"Exp105 Cost did not reproduce: {payload['cost_mae']} vs expected {expected_cost}")
    if payload["delay_mape_percent"] is None or not np.isfinite(float(payload["delay_mape_percent"])):
        raise RuntimeError("Delay percentage error (MAPE) was not finite")
    if not _close(payload["persisted_inference_cost_mae"], payload["cost_mae"]):
        raise RuntimeError(
            "Persisted/live Exp105 Cost inference does not match the verified in-memory prediction: "
            f"{payload['persisted_inference_cost_mae']} vs {payload['cost_mae']}"
        )
    if not _close(payload["persisted_inference_delay_mae"], payload["delay_mae"]):
        raise RuntimeError(
            "Persisted/live Exp113 Delay inference does not match the verified in-memory prediction: "
            f"{payload['persisted_inference_delay_mae']} vs {payload['delay_mae']}"
        )
    if a.end == 2021:
        if payload["cost_improvement_percentage"] <= 0:
            raise RuntimeError("Exp105 Cost production promotion did not improve Cost")
        if payload["delay_improvement_percentage"] <= 0:
            raise RuntimeError("Exp113 Delay production promotion did not improve Delay")
        if payload["comparison_test_projects"] < 2 or payload["comparison_test_snapshots"] < payload["comparison_test_projects"]:
            raise RuntimeError("Evidence-defined comparison cohort is not usable")
    if payload["risk_retained"] is not True:
        raise RuntimeError("Combined production promotion failed Risk isolation guard")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    prefix = f"EXP105_EXP113_PRODUCTION_{a.start}_{a.end}"
    print(f"{prefix}_COST_MAE={payload['cost_mae']}")
    print(f"{prefix}_COST_IMPROVEMENT_PERCENT={payload['cost_improvement_percentage']}")
    print(f"{prefix}_DELAY_MAE={payload['delay_mae']}")
    print(f"{prefix}_DELAY_MAPE_PERCENT={payload['delay_mape_percent']}")
    print(f"{prefix}_DELAY_IMPROVEMENT_PERCENT={payload['delay_improvement_percentage']}")
    print(f"{prefix}_PROJECTS={payload['comparison_test_projects']}")
    print(f"{prefix}_SNAPSHOTS={payload['comparison_test_snapshots']}")


if __name__ == "__main__":
    main()
