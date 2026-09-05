from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.experiments.exp94_family_disagreement_cost import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, required=True, choices=(2019, 2021))
    args = parser.parse_args()

    out = Path("test-output/exp94-family-disagreement") / f"2001_{args.end}"
    out.mkdir(parents=True, exist_ok=True)
    result = run_experiment(args.end, out / "baseline")
    path = out / "experiment_result.json"
    path.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
