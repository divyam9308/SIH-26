"""Temporary cost-only fresh audit for three current 25-feature training windows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
    training_as_of_invariants,
)
from backend.app.ml.monthly_training import _regression_metrics, _select_regressor
from backend.app.ml.provenance import feature_schema_fingerprint, frame_fingerprint, git_commit_sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_outputs" / "fresh_three_window_cost_25f.json"
WINDOWS = [(2001, 2019), (2001, 2021), (2018, 2021)]
TEST_START, TEST_END = 2022, 2025


def main() -> None:
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    test = data[data["completion_year"].between(TEST_START, TEST_END)].copy()
    if not training_as_of_invariants(test)["passed"]:
        raise RuntimeError("Common holdout failed as-of invariants")

    results = []
    canonical_features = None
    for start, end in WINDOWS:
        train = data[data["completion_year"].between(start, end)].copy()
        overlap = set(train.canonical_project_id.dropna()) & set(test.canonical_project_id.dropna())
        if overlap:
            raise RuntimeError(f"Project leakage for {start}-{end}: {len(overlap)}")
        if not training_as_of_invariants(train)["passed"]:
            raise RuntimeError(f"Training as-of invariants failed for {start}-{end}")

        audit = audit_features(
            train,
            CANDIDATE_FEATURES,
            minimum_availability=10,
            minimum_year_coverage=2,
            as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
            leakage_risks={
                "revised_cost_cr": "same-snapshot late-stage signal",
                "cost_escalation_percentage": "same-snapshot late-stage signal",
            },
        )
        features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
        if len(features) != 25:
            raise RuntimeError(f"Expected 25 features for {start}-{end}, got {len(features)}")
        if canonical_features is None:
            canonical_features = features
        elif features != canonical_features:
            raise RuntimeError(f"Feature schema differs for {start}-{end}")

        algorithm, model, internal = _select_regressor(
            train, features, "actual_cost_overrun_percentage", 26203
        )
        predictions = np.asarray(model.predict(test[features]), dtype=float)
        metrics = _regression_metrics(
            test.actual_cost_overrun_percentage,
            predictions,
            test.sample_weight,
            test.canonical_project_id,
        )
        results.append({
            "window": f"{start}_{end}",
            "training_period": [start, end],
            "testing_period": [TEST_START, TEST_END],
            "training_rows": int(len(train)),
            "training_projects": int(train.canonical_project_id.nunique()),
            "test_rows": int(len(test)),
            "test_projects": int(test.canonical_project_id.nunique()),
            "feature_count": len(features),
            "feature_schema_fingerprint": feature_schema_fingerprint(features),
            "training_fingerprint": frame_fingerprint(train),
            "selected_cost_algorithm": algorithm,
            "internal_cost_algorithm_comparison": internal,
            "cost_metrics": metrics,
        })

    payload = {
        "audit": "fresh_current_25_feature_cost_three_window_common_holdout",
        "source_commit": git_commit_sha(ROOT),
        "dataset_fingerprint": frame_fingerprint(data),
        "identity_rows": int(len(identity)),
        "dataset_rows": int(len(data)),
        "dataset_projects": int(data.canonical_project_id.nunique()),
        "common_test_period": [TEST_START, TEST_END],
        "common_test_fingerprint": frame_fingerprint(test),
        "common_test_rows": int(len(test)),
        "common_test_projects": int(test.canonical_project_id.nunique()),
        "feature_count": len(canonical_features or []),
        "features": canonical_features,
        "feature_schema_fingerprint": feature_schema_fingerprint(canonical_features or []),
        "results": results,
        "notes": [
            "Fresh fit only: saved evaluation_results.json metrics are never read.",
            "All windows use the identical 2022-2025 holdout and identical 25-feature schema.",
            "Cost estimator family is freshly selected using the current production temporal-selection code for each window.",
            "No production model artifacts are written.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("FRESH_COST_AUDIT=" + json.dumps({
        item["window"]: {
            "algorithm": item["selected_cost_algorithm"],
            **item["cost_metrics"],
            "training_projects": item["training_projects"],
            "test_projects": item["test_projects"],
        }
        for item in results
    }, sort_keys=True))


if __name__ == "__main__":
    main()
