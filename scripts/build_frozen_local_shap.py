"""Create verified local-SHAP entries without training or modifying model artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.services.frozen_explanation_service import build_local_explanation


parser = argparse.ArgumentParser()
parser.add_argument("--window", required=True)
parser.add_argument("--project", required=True)
args = parser.parse_args()
print(json.dumps(build_local_explanation(args.window, args.project), indent=2))
