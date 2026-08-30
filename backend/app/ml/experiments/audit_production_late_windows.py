"""One-off production audit for later temporal training cutoffs.

This is intentionally not a reusable experiment adapter. It freshly retrains the
current Exp61 production stack in an isolated temporary artifact root and reports
Cost/Delay MAE for three fixed windows only.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.nextgen_common import _hash_prod, _prepare, normalize_taxonomy
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _json_safe, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp61_baseline import (
    PRODUCTION_COST_BASELINE,
    PRODUCTION_DELAY_BASELINE,
    train_window_with_promoted_cost_and_delay,
)

TRAINING_START = 2001
TEST_END = 2025
WINDOWS = {
    2021: (2022, 2025),
    2022: (2023, 2025),
    2023: (2024, 2025),
}


def window_contract(training_end: int) -> tuple[int, int]:
    if training_end not in WINDOWS:
        raise ValueError(f"Supported one-off audit cutoffs are {sorted(WINDOWS)}")
    return WINDOWS[training_end]


def run_audit(training_end: int) -> dict:
    expected_test_start, expected_test_end = window_contract(training_end)
    before = _hash_prod()
    data, identity = build_training_dataset()

    # Independently verify the exact future cohort implied by the requested cutoff.
    prepared = normalize_taxonomy(_prepare(data))
    _, test = temporal_project_split(prepared, TRAINING_START, training_end, TEST_END)
    cohort = _production_cost_evaluation_rows(test).copy()
    observed_years = sorted(
        pd.to_numeric(cohort["completion_year"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    if not observed_years:
        raise RuntimeError("Production audit produced an empty future cohort")
    if min(observed_years) < expected_test_start or max(observed_years) > expected_test_end:
        raise RuntimeError(
            f"Unexpected holdout years {observed_years}; expected only {expected_test_start}-{expected_test_end}"
        )

    with tempfile.TemporaryDirectory(prefix=f"production-audit-{training_end}-") as td:
        root = Path(td) / "production"
        receipt = train_window_with_promoted_cost_and_delay(
            TRAINING_START,
            training_end,
            TEST_END,
            data=data,
            identity=identity,
            artifact_root=root,
        )

    if before != _hash_prod():
        raise AssertionError("One-off audit modified tracked production artifacts")

    lifecycle_metrics = receipt["lifecycle"]["metrics"]
    metadata = receipt["metadata"]
    cost_mae = float(lifecycle_metrics["cost"]["MAE"])
    delay_mae = float(lifecycle_metrics["delay"]["MAE"])

    return {
        "audit_type": "one_off_current_production_temporal_window_audit",
        "training_start": TRAINING_START,
        "training_end": training_end,
        "test_start": expected_test_start,
        "test_end": expected_test_end,
        "observed_test_completion_years": observed_years,
        "production_cost_baseline": metadata.get("production_cost_baseline", PRODUCTION_COST_BASELINE),
        "production_delay_baseline": metadata.get("production_delay_baseline", PRODUCTION_DELAY_BASELINE),
        "production_cost_mae": cost_mae,
        "production_delay_mae": delay_mae,
        "comparison_test_projects": int(cohort["canonical_project_id"].nunique()),
        "comparison_test_snapshots": int(len(cohort)),
        "production_artifacts_untouched": True,
        "future_experiment_harness": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = run_audit(args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")

    prefix = f"PRODUCTION_AUDIT_2001_{args.end}"
    print(f"{prefix}_TEST={payload['test_start']}_{payload['test_end']}")
    print(f"{prefix}_COST_MAE={payload['production_cost_mae']}")
    print(f"{prefix}_DELAY_MAE={payload['production_delay_mae']}")
    print(f"{prefix}_PROJECTS={payload['comparison_test_projects']}")
    print(f"{prefix}_SNAPSHOTS={payload['comparison_test_snapshots']}")


if __name__ == "__main__":
    main()
