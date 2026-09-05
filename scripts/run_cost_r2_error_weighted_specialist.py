from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.experiments.cost_r2_error_weighted_specialist import run_experiment

OUT = Path("test-output/cost-r2-error-weighted-specialist")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_experiment(OUT / "baseline")
    path = OUT / "experiment_result.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
