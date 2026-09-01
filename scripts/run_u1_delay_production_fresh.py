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

EXPECTED = {
    2019: {"cost": 27.801, "delay": 438.098},
    2021: {"cost": 25.829, "delay": 346.599},
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
    if a.start != 2001 or a.end not in EXPECTED or a.test_end != 2025:
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
        "persisted_inference_delay_mae": live_delay_mae,
        "delay_improvement_percentage": promo["delay"]["delay_improvement_percentage"],
        "comparison_test_projects": contract["test_projects"],
        "comparison_test_snapshots": contract["test_snapshots"],
        "production_cost_baseline": result["metadata"]["production_cost_baseline"],
        "production_delay_baseline": result["metadata"]["production_delay_baseline"],
        "promoted_cost_from_experiment": result["metadata"]["promoted_cost_from_experiment"],
        "promoted_delay_from_experiment": result["metadata"]["promoted_delay_from_experiment"],
        "risk_retained": promo["risk_retained"],
    }

    expected = EXPECTED[a.end]
    if not _close(payload["cost_mae"], expected["cost"]):
        raise RuntimeError(f"Exp105 Cost did not reproduce: {payload['cost_mae']} vs expected {expected['cost']}")
    if not _close(payload["delay_mae"], expected["delay"]):
        raise RuntimeError(f"Exp113 Delay did not reproduce: {payload['delay_mae']} vs expected {expected['delay']}")
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
        if (payload["comparison_test_projects"], payload["comparison_test_snapshots"]) != (721, 11200):
            raise RuntimeError("Verified 2001-2021 production cohort changed")
    if payload["risk_retained"] is not True:
        raise RuntimeError("Combined production promotion failed Risk isolation guard")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    prefix = f"EXP105_EXP113_PRODUCTION_{a.start}_{a.end}"
    print(f"{prefix}_COST_MAE={payload['cost_mae']}")
    print(f"{prefix}_COST_IMPROVEMENT_PERCENT={payload['cost_improvement_percentage']}")
    print(f"{prefix}_DELAY_MAE={payload['delay_mae']}")
    print(f"{prefix}_DELAY_IMPROVEMENT_PERCENT={payload['delay_improvement_percentage']}")
    print(f"{prefix}_PROJECTS={payload['comparison_test_projects']}")
    print(f"{prefix}_SNAPSHOTS={payload['comparison_test_snapshots']}")


if __name__ == "__main__":
    main()
