"""Temporary read-only audit: freshly train current lifecycle model on three windows.

This script is intentionally audit-only. It does not write production model artifacts.
All three models are evaluated on the identical 2022-2025 holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
    training_as_of_invariants,
)
from backend.app.ml.monthly_training import (
    _balanced_stage_summary,
    _stage_metrics,
    _train_variant,
)
from backend.app.ml.provenance import (
    feature_schema_fingerprint,
    frame_fingerprint,
    git_commit_sha,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_outputs" / "fresh_three_window_25f.json"
WINDOWS = [(2001, 2019), (2001, 2021), (2018, 2021)]
TEST_START = 2022
TEST_END = 2025


def project_weight_check(frame: pd.DataFrame) -> dict:
    sums = frame.groupby("canonical_project_id", dropna=False)["sample_weight"].sum()
    deviation = (sums - 1.0).abs()
    return {
        "projects": int(len(sums)),
        "max_abs_deviation_from_one": float(deviation.max()) if len(deviation) else None,
        "passed": bool(len(deviation) and deviation.max() < 1e-9),
    }


def main() -> None:
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")

    test = data[data["completion_year"].between(TEST_START, TEST_END)].copy()
    if test["canonical_project_id"].nunique() < 2:
        raise RuntimeError("Common 2022-2025 holdout is unexpectedly empty.")
    test_invariants = training_as_of_invariants(test)
    if not test_invariants["passed"]:
        raise RuntimeError(f"Holdout as-of invariant failure: {test_invariants}")

    results = []
    canonical_features = None
    canonical_feature_fp = None

    for start, end in WINDOWS:
        train = data[data["completion_year"].between(start, end)].copy()
        overlap = set(train["canonical_project_id"].dropna()) & set(test["canonical_project_id"].dropna())
        if overlap:
            raise RuntimeError(f"Project leakage for {start}-{end}: {len(overlap)} overlapping projects")

        train_invariants = training_as_of_invariants(train)
        if not train_invariants["passed"]:
            raise RuntimeError(f"Training as-of invariant failure for {start}-{end}: {train_invariants}")

        audit = audit_features(
            train,
            CANDIDATE_FEATURES,
            minimum_availability=10,
            minimum_year_coverage=2,
            as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
            leakage_risks={
                "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by ablation",
                "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by ablation",
            },
        )
        features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
        feature_fp = feature_schema_fingerprint(features)
        if len(features) != 25:
            raise RuntimeError(f"Expected current 25-feature lifecycle contract for {start}-{end}, got {len(features)}: {features}")
        if canonical_features is None:
            canonical_features = features
            canonical_feature_fp = feature_fp
        elif features != canonical_features:
            raise RuntimeError(
                f"Feature contract changed across windows. Canonical={canonical_features}; {start}-{end}={features}"
            )

        bundle, metrics, rows = _train_variant(train, test, features, 26203)
        stages = _stage_metrics(rows)
        balanced = _balanced_stage_summary(stages)

        results.append({
            "window": f"{start}_{end}",
            "training_period": [start, end],
            "testing_period": [TEST_START, TEST_END],
            "training_rows": int(len(train)),
            "training_projects": int(train["canonical_project_id"].nunique()),
            "test_rows": int(len(test)),
            "test_projects": int(test["canonical_project_id"].nunique()),
            "feature_count": len(features),
            "features": features,
            "feature_schema_fingerprint": feature_fp,
            "training_fingerprint": frame_fingerprint(train),
            "selected_algorithms": bundle["selected_algorithms"],
            "internal_comparisons": bundle["internal_comparisons"],
            "metrics": metrics,
            "lifecycle_stage_metrics": stages,
            "balanced_stage_summary": balanced,
            "training_as_of_invariants": train_invariants,
            "training_project_weight_check": project_weight_check(train),
        })

    payload = {
        "audit": "fresh_current_25_feature_three_window_common_holdout",
        "source_commit": git_commit_sha(ROOT),
        "dataset_fingerprint": frame_fingerprint(data),
        "identity_rows": int(len(identity)),
        "dataset_rows": int(len(data)),
        "dataset_projects": int(data["canonical_project_id"].nunique()),
        "common_test_period": [TEST_START, TEST_END],
        "common_test_fingerprint": frame_fingerprint(test),
        "common_test_rows": int(len(test)),
        "common_test_projects": int(test["canonical_project_id"].nunique()),
        "common_test_as_of_invariants": test_invariants,
        "common_test_project_weight_check": project_weight_check(test),
        "feature_count": len(canonical_features or []),
        "features": canonical_features,
        "feature_schema_fingerprint": canonical_feature_fp,
        "results": results,
        "notes": [
            "All models were freshly fit in this workflow run; no saved evaluation_results.json metrics were used.",
            "All three windows use the identical 2022-2025 holdout and identical 25-feature schema.",
            "Current production training code selects the cost and delay regressor family independently inside each training window using its latest training year as temporal validation.",
            "No production artifact directory is written by this audit.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    print("FRESH_THREE_WINDOW_AUDIT=" + json.dumps({
        item["window"]: {
            "cost_mae": item["metrics"]["cost"]["MAE"],
            "cost_rmse": item["metrics"]["cost"]["RMSE"],
            "cost_r2": item["metrics"]["cost"]["R2"],
            "cost_mape": item["metrics"]["cost"]["MAPE"],
            "delay_mae": item["metrics"]["delay"]["MAE"],
            "risk_macro_f1": item["metrics"]["risk"]["macro_f1"],
            "cost_algorithm": item["selected_algorithms"]["cost"],
            "delay_algorithm": item["selected_algorithms"]["delay"],
            "training_projects": item["training_projects"],
            "test_projects": item["test_projects"],
        }
        for item in results
    }, sort_keys=True))


if __name__ == "__main__":
    main()
