from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.experiments import cost_r2_error_weighted_specialist as experiment
from backend.app.ml.experiments.cost_r2_oof_sharding import (
    OOF_YEARS,
    generate_oof_shard,
    load_oof_shards,
    validate_oof_against_context,
)

OUT = Path("test-output/cost-r2-error-weighted-specialist")


def _write_result(result: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "experiment_result.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


def _run_with_precomputed_oof(oof_dir: Path) -> dict:
    production_oof = load_oof_shards(oof_dir)
    original = experiment.strict_production_oof

    def precomputed(ctx: dict, max_folds: int = 4):
        return validate_oof_against_context(
            production_oof,
            ctx["train"],
            experiment.forward_folds,
            max_folds=max_folds,
        )

    experiment.strict_production_oof = precomputed
    try:
        return experiment.run_experiment(OUT / "baseline")
    finally:
        experiment.strict_production_oof = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-year", type=int, choices=OOF_YEARS)
    parser.add_argument("--oof-dir", type=Path)
    args = parser.parse_args()

    if args.oof_year is not None and args.oof_dir is not None:
        parser.error("--oof-year and --oof-dir are mutually exclusive")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.oof_year is not None:
        path = OUT / "oof-shards" / f"production_oof_{args.oof_year}.pkl"
        metadata = generate_oof_shard(
            year=args.oof_year,
            forward_folds=experiment.forward_folds,
            fold_trainer=experiment._strict_production_oof_fold,
            output_path=path,
        )
        print(json.dumps({"strict_production_oof_shard": metadata}, indent=2))
        return

    result = (
        _run_with_precomputed_oof(args.oof_dir)
        if args.oof_dir is not None
        else experiment.run_experiment(OUT / "baseline")
    )
    _write_result(result)


if __name__ == "__main__":
    main()
