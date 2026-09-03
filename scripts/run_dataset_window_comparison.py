from __future__ import annotations

import argparse
from pathlib import Path

from backend.app.ml.experiments.dataset_window_comparison import WINDOWS, run_window, write_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one current-production dataset-window comparison")
    parser.add_argument("--window", required=True, choices=sorted(WINDOWS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_window(args.window)
    output = args.output or Path("reports/dataset_window_comparison") / f"{args.window}.json"
    write_result(result, output)

    print(f"WINDOW={result['label']}")
    print(f"TRAIN={result['training_window']}")
    print(f"TEST={result['test_window']}")
    print(f"COST_MAE={result['cost_mae']:.6f}")
    print(f"DELAY_MAE_DAYS={result['delay_mae_days']:.6f}")
    print(f"COMPARISON_PROJECTS={result['comparison_projects']}")
    print(f"COMPARISON_SNAPSHOTS={result['comparison_snapshots']}")
    print(f"RESULT_JSON={output}")


if __name__ == "__main__":
    main()
