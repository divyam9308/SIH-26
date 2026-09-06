"""Run Exp135 on one canonical historical window without touching production artifacts."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from backend.app.ml.experiments.exp135_log_space_cost import train_window_with_exp135
from backend.app.ml.monthly_lifecycle import build_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data, identity = build_training_dataset()
    with tempfile.TemporaryDirectory(prefix=f"exp135-{args.end}-") as td:
        result = train_window_with_exp135(
            2001,
            args.end,
            2025,
            data=data,
            identity=identity,
            artifact_root=Path(td) / "models",
        )
    payload = result["exp135"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
