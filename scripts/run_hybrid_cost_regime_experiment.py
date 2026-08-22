#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml.experiments.hybrid_cost_regime import run_experiment


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
