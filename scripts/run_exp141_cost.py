from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.ml.experiments.exp141_stage_conditional_cost import fit_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp141 Lifecycle-Stage Conditional Cost Challenger")
    parser.add_argument("--end", type=int, choices=[2021, 2022], default=2022)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fit_experiment(training_start=2001, training_end=args.end, output=args.output)


if __name__ == "__main__":
    main()
