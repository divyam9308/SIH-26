"""Freshly retrain exact production, then evaluate isolated Experiment 45."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.production_exp35_baseline import train_window_with_promoted_cost_and_delay

ROOT = Path(__file__).resolve().parents[1]


def _production_hashes() -> dict[str, str]:
    root = ROOT / "models" / "monthly_lifecycle"
    result = {}
    if not root.exists():
        return result
    for path in sorted(root.glob("*/*")):
        if path.is_file() and "experiments" not in path.parts:
            result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--test-end", default=2025, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    before = _production_hashes()
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    with tempfile.TemporaryDirectory(prefix="exp45-production-") as temp:
        artifact_root = Path(temp) / "production"
        production_result = train_window_with_promoted_cost_and_delay(
            args.start, args.end, args.test_end, data=data, identity=identity, artifact_root=artifact_root
        )
        target = artifact_root / f"{args.start}_{args.end}"
        bundle = {
            "cost": joblib.load(target / "cost_model.pkl"),
            "delay": joblib.load(target / "delay_model.pkl"),
            "metadata": json.loads((target / "metadata.json").read_text()),
        }
        fitted = get_experiment_adapter("exp_45").module.fit_against_production(
            data=data,
            training_start=args.start,
            training_end=args.end,
            test_end=args.test_end,
            production_bundle=bundle,
            production_receipt=production_result,
        )

    if before != _production_hashes():
        raise AssertionError("Exp45 modified a production artifact")
    overall = dict(fitted["overall_comparison"])
    if args.start == 2001 and args.end == 2021 and args.test_end == 2025:
        expected = {"projects": 721, "snapshots": 11200, "cost": 26.287, "delay": 431.618}
        observed = {
            "projects": overall["comparison_test_projects"],
            "snapshots": overall["comparison_test_snapshots"],
            "cost": float(overall["production_cost_mae"]),
            "delay": float(overall["production_delay_mae"]),
        }
        valid = (
            observed["projects"] == expected["projects"]
            and observed["snapshots"] == expected["snapshots"]
            and abs(observed["cost"] - expected["cost"]) <= 0.005
            and abs(observed["delay"] - expected["delay"]) <= 0.005
        )
        if not valid:
            print("EXP45_EXECUTION_VERDICT=EXECUTION INVALID")
            raise RuntimeError(f"Exp45 freshly reproduced baseline violates contract: {observed}")

    payload = {
        "window": f"{args.start}_{args.end}",
        "test_end": args.test_end,
        "baseline_mode": "fresh exact production Exp35 retraining in an isolated temporary artifact root",
        "production": production_result,
        "experiment": fitted["experiment"],
        "overall_comparison": overall,
        "production_artifacts_untouched": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")
    prefix = f"EXP45_{args.start}_{args.end}"
    print(f"{prefix}_PRODUCTION_COST_MAE={overall['production_cost_mae']}")
    print(f"{prefix}_EXPERIMENT_COST_MAE={overall['experiment_cost_mae']}")
    print(f"{prefix}_COST_IMPROVEMENT_PERCENT={overall['cost_improvement_percentage']}")
    print("EXP45_EXECUTION_VERDICT=EXECUTION VALID")
    print(f"EXP45_SCIENTIFIC_VERDICT={overall['scientific_verdict']}")


if __name__ == "__main__":
    main()
