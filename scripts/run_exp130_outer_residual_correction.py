from __future__ import annotations

import argparse
from backend.app.ml.experiments.exp130_outer_residual_correction import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp130 outer production residual correction")
    parser.add_argument(
        "--output",
        default="reports/exp130_outer_residual_2001_2021.json",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
