"""Read-only cost audit for Experiment 3 across one historical cutoff.

This script intentionally writes only under reports/audits. It does not use the
production model artifact directory. It compares:
1) the current lifecycle production cost path on its full eligible cohort; and
2) direct-vs-residual final-overrun forecasts on Experiment 3's exact common cohort.
"""
from __future__ import annotations

import argparse
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
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    _select_regressor,
    temporal_project_split,
)
from backend.app.ml.provenance import frame_fingerprint
from backend.app.ml.residual_overrun_experiment import (
    CURRENT_OVERRUN,
    FINAL_TARGET,
    RESIDUAL_TARGET,
    _renormalize_project_weights,
    _with_residual_target,
    prepare_common_cost_cohort,
    reconstruct_final_overrun,
    run_residual_overrun_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audits"
STAGES = ("early", "mid", "late", "very_late")


def lifecycle_feature_audit(train: pd.DataFrame) -> dict:
    return audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by production ablation",
            "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by production ablation",
        },
    )


def lifecycle_features(audit: dict) -> list[str]:
    return list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))


def weight_summary(frame: pd.DataFrame) -> dict:
    totals = frame.groupby("canonical_project_id")["sample_weight"].sum().astype(float)
    return {
        "projects": int(len(totals)),
        "minimum": round(float(totals.min()), 9),
        "median": round(float(totals.median()), 9),
        "maximum": round(float(totals.max()), 9),
        "all_close_to_one": bool(np.allclose(totals.to_numpy(), 1.0, atol=1e-8, rtol=0)),
    }


def stage_cost_metrics(rows: pd.DataFrame, prediction: str) -> dict:
    result = {}
    for stage in STAGES:
        part = rows[rows["lifecycle_stage"].eq(stage)]
        if part.empty:
            result[stage] = {"available": False}
            continue
        result[stage] = {
            "available": True,
            "metrics": _regression_metrics(
                part[FINAL_TARGET],
                part[prediction].to_numpy(dtype=float),
                part["sample_weight"],
                part["canonical_project_id"],
            ),
        }
    return result


def bootstrap_difference(rows: pd.DataFrame, *, seed: int = 73103, samples: int = 1000) -> dict:
    temp = rows[["canonical_project_id", "sample_weight", "direct_abs_error", "residual_abs_error"]].copy()
    projects = []
    for project_id, group in temp.groupby("canonical_project_id", sort=False):
        w = group["sample_weight"].to_numpy(dtype=float)
        total = float(w.sum())
        if total <= 0:
            continue
        projects.append((
            str(project_id),
            float(np.average(group["direct_abs_error"], weights=w)),
            float(np.average(group["residual_abs_error"], weights=w)),
        ))
    if len(projects) < 2:
        return {"available": False, "reason": "fewer than two projects"}
    arr = np.asarray([[d, r] for _, d, r in projects], dtype=float)
    delta = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, len(arr), size=len(arr))
        boot[i] = float(np.mean(delta[idx]))
    low, high = np.quantile(boot, [0.025, 0.975])
    observed = float(np.mean(delta))
    verdict = "inconclusive"
    if low > 0:
        verdict = "residual_better"
    elif high < 0:
        verdict = "residual_worse"
    return {
        "available": True,
        "samples": samples,
        "seed": seed,
        "projects": len(projects),
        "metric": "direct_project_balanced_MAE_minus_residual_project_balanced_MAE",
        "observed_difference": round(observed, 6),
        "ci95": [round(float(low), 6), round(float(high), 6)],
        "verdict": verdict,
    }


def project_diagnostics(rows: pd.DataFrame, tie_pp: float = 0.5) -> dict:
    records = []
    for project_id, group in rows.groupby("canonical_project_id", sort=False):
        w = group["sample_weight"].to_numpy(dtype=float)
        direct = float(np.average(group["direct_abs_error"], weights=w))
        residual = float(np.average(group["residual_abs_error"], weights=w))
        records.append((str(project_id), direct, residual))
    arr = np.asarray([[d, r] for _, d, r in records], dtype=float)
    diff = arr[:, 0] - arr[:, 1]
    tied = np.abs(diff) <= tie_pp
    residual_better = diff > tie_pp
    direct_better = diff < -tie_pp
    return {
        "projects": int(len(arr)),
        "tie_definition_pp": tie_pp,
        "residual_better_percentage": round(float(residual_better.mean() * 100), 3),
        "direct_better_percentage": round(float(direct_better.mean() * 100), 3),
        "approximately_tied_percentage": round(float(tied.mean() * 100), 3),
        "median_direct_absolute_error": round(float(np.median(arr[:, 0])), 3),
        "median_residual_absolute_error": round(float(np.median(arr[:, 1])), 3),
        "direct_p75": round(float(np.quantile(arr[:, 0], .75)), 3),
        "residual_p75": round(float(np.quantile(arr[:, 1], .75)), 3),
        "direct_p90": round(float(np.quantile(arr[:, 0], .90)), 3),
        "residual_p90": round(float(np.quantile(arr[:, 1], .90)), 3),
    }


def controlled_e3(data: pd.DataFrame, start: int, end: int, test_end: int, common_start: int) -> dict:
    train, natural_test = prepare_common_cost_cohort(data, start, end, test_end)
    audit = lifecycle_feature_audit(train)
    features = lifecycle_features(audit)
    algorithm, direct_model, internal = _select_regressor(train, features, FINAL_TARGET, 27103)
    residual_model = _fit_pipeline(_regressors(27103)[algorithm], train, features, RESIDUAL_TARGET)

    def evaluate(test: pd.DataFrame) -> dict:
        direct = np.asarray(direct_model.predict(test[features]), dtype=float)
        remaining = np.asarray(residual_model.predict(test[features]), dtype=float)
        residual_final = reconstruct_final_overrun(test[CURRENT_OVERRUN], remaining)
        direct_metrics = _regression_metrics(test[FINAL_TARGET], direct, test.sample_weight, test.canonical_project_id)
        residual_metrics = _regression_metrics(test[FINAL_TARGET], residual_final, test.sample_weight, test.canonical_project_id)
        residual_target_metrics = _regression_metrics(test[RESIDUAL_TARGET], remaining, test.sample_weight, test.canonical_project_id)
        d_mae = direct_metrics["MAE"]
        r_mae = residual_metrics["MAE"]
        improvement = round((d_mae - r_mae) / d_mae * 100, 3) if d_mae not in (None, 0) else None
        rows = test[["canonical_project_id", "lifecycle_stage", FINAL_TARGET, "sample_weight"]].copy()
        rows["direct_prediction"] = direct
        rows["residual_prediction"] = residual_final
        rows["direct_abs_error"] = np.abs(direct - rows[FINAL_TARGET].to_numpy(dtype=float))
        rows["residual_abs_error"] = np.abs(residual_final - rows[FINAL_TARGET].to_numpy(dtype=float))
        return {
            "rows": int(len(test)),
            "projects": int(test.canonical_project_id.nunique()),
            "weight_summary": weight_summary(test),
            "direct_metrics": direct_metrics,
            "residual_final_metrics": residual_metrics,
            "residual_target_metrics_diagnostic_only": residual_target_metrics,
            "improvement_percentage": improvement,
            "threshold_10pct_passed": bool(improvement is not None and improvement >= 10.0),
            "direct_stage_metrics": stage_cost_metrics(rows, "direct_prediction"),
            "residual_stage_metrics": stage_cost_metrics(rows, "residual_prediction"),
            "project_diagnostics": project_diagnostics(rows),
            "bootstrap": bootstrap_difference(rows),
        }

    natural = evaluate(natural_test)
    common_test = natural_test[natural_test["completion_year"].between(common_start, test_end)].copy()
    if common_test.empty:
        common = {"available": False, "reason": "no common-holdout rows"}
    else:
        # completion_year is project-constant, so this retains complete project snapshot sets.
        common = {"available": True, **evaluate(common_test)}
    return {
        "features": features,
        "feature_count": len(features),
        "feature_audit": {
            "data_quality_score": audit["data_quality_score"],
            "as_of_evidence_coverage": audit["as_of_evidence_coverage"],
            "removed_features": audit["removed_features"],
        },
        "selected_algorithm": algorithm,
        "internal_algorithm_comparison": internal,
        "training_rows": int(len(train)),
        "training_projects": int(train.canonical_project_id.nunique()),
        "training_weight_summary": weight_summary(train),
        "controls": {
            "same_train_snapshots": True,
            "same_test_snapshots": True,
            "same_features": True,
            "same_regressor_family": True,
            "post_sampling_project_weights": True,
            "common_cohort_weights_renormalized": True,
            "explicit_as_of_lineage": True,
        },
        "natural": natural,
        "common_2022_plus": common,
    }


def production_cost(data: pd.DataFrame, start: int, end: int, test_end: int, common_start: int) -> dict:
    train, natural_test = temporal_project_split(data, start, end, test_end)
    train_inv = training_as_of_invariants(train)
    test_inv = training_as_of_invariants(natural_test)
    if not train_inv["passed"] or not test_inv["passed"]:
        raise RuntimeError(f"as-of invariant failure: train={train_inv}, test={test_inv}")
    audit = lifecycle_feature_audit(train)
    features = lifecycle_features(audit)
    algorithm, model, internal = _select_regressor(train, features, FINAL_TARGET, 26203)

    def evaluate(test: pd.DataFrame) -> dict:
        pred = np.asarray(model.predict(test[features]), dtype=float)
        metrics = _regression_metrics(test[FINAL_TARGET], pred, test.sample_weight, test.canonical_project_id)
        rows = test[["canonical_project_id", "lifecycle_stage", FINAL_TARGET, "sample_weight"]].copy()
        rows["prediction"] = pred
        return {
            "rows": int(len(test)),
            "projects": int(test.canonical_project_id.nunique()),
            "metrics": metrics,
            "stage_metrics": stage_cost_metrics(rows, "prediction"),
            "weight_summary": weight_summary(test),
        }

    natural = evaluate(natural_test)
    common_test = natural_test[natural_test["completion_year"].between(common_start, test_end)].copy()
    common = {"available": False, "reason": "no common-holdout rows"} if common_test.empty else {"available": True, **evaluate(common_test)}
    return {
        "features": features,
        "feature_count": len(features),
        "feature_audit": {
            "data_quality_score": audit["data_quality_score"],
            "as_of_evidence_coverage": audit["as_of_evidence_coverage"],
            "removed_features": audit["removed_features"],
        },
        "selected_algorithm": algorithm,
        "internal_algorithm_comparison": internal,
        "training_rows": int(len(train)),
        "training_projects": int(train.canonical_project_id.nunique()),
        "training_weight_summary": weight_summary(train),
        "natural": natural,
        "common_2022_plus": common,
        "as_of_invariants": {"training": train_inv, "testing": test_inv},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", required=True, type=int)
    parser.add_argument("--training-start", default=2001, type=int)
    parser.add_argument("--common-start", default=2022, type=int)
    args = parser.parse_args()

    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    years = data["completion_year"].dropna().astype(int)
    latest = int(years.max())
    start, end = int(args.training_start), int(args.training_end)
    if end >= latest:
        raise ValueError(f"training end {end} must precede latest completion year {latest}")

    production = production_cost(data, start, end, latest, args.common_start)
    e3 = controlled_e3(data, start, end, latest, args.common_start)

    sanity = None
    if start == 2001 and end == 2015:
        official = run_residual_overrun_experiment(start, end, test_end=latest, data=data, persist=False)
        sanity = {
            "official_direct_mae": official["direct_final_overrun_metrics"]["MAE"],
            "harness_direct_mae": e3["natural"]["direct_metrics"]["MAE"],
            "official_residual_mae": official["residual_reconstructed_final_overrun_metrics"]["MAE"],
            "harness_residual_mae": e3["natural"]["residual_final_metrics"]["MAE"],
            "official_improvement_percentage": official["final_mae_improvement_percentage"],
            "harness_improvement_percentage": e3["natural"]["improvement_percentage"],
        }
        sanity["exact_match"] = bool(
            sanity["official_direct_mae"] == sanity["harness_direct_mae"]
            and sanity["official_residual_mae"] == sanity["harness_residual_mae"]
            and sanity["official_improvement_percentage"] == sanity["harness_improvement_percentage"]
        )
        if not sanity["exact_match"]:
            raise RuntimeError(f"audit harness disagrees with official Experiment 3: {sanity}")

    result = {
        "audit": "experiment3_multiwindow_cost",
        "training_period": [start, end],
        "natural_testing_period": [end + 1, latest],
        "common_testing_period": [args.common_start, latest],
        "dataset": {
            "fingerprint": frame_fingerprint(data),
            "rows": int(len(data)),
            "projects": int(data.canonical_project_id.nunique()),
            "completion_year_range": [int(years.min()), latest],
            "identity_rows": int(len(identity)),
            "identity_verified_rows": int(identity.identity_verified.sum()),
        },
        "production_full_cohort": production,
        "experiment3_controlled_common_cohort": e3,
        "official_2001_2015_sanity_check": sanity,
        "cohort_warning": "Production full-cohort MAE and Experiment 3 MAEs use different eligibility cohorts and must not be treated as a head-to-head target-formulation comparison.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"experiment3_cost_audit_2001_{end}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    summary = {
        "training_end": end,
        "latest_year": latest,
        "dataset_fingerprint": result["dataset"]["fingerprint"],
        "production_natural_mae": production["natural"]["metrics"]["MAE"],
        "production_common_mae": production["common_2022_plus"].get("metrics", {}).get("MAE"),
        "e3_direct_natural_mae": e3["natural"]["direct_metrics"]["MAE"],
        "e3_residual_natural_mae": e3["natural"]["residual_final_metrics"]["MAE"],
        "e3_natural_improvement_pct": e3["natural"]["improvement_percentage"],
        "e3_natural_bootstrap": e3["natural"]["bootstrap"],
        "e3_direct_common_mae": e3["common_2022_plus"].get("direct_metrics", {}).get("MAE"),
        "e3_residual_common_mae": e3["common_2022_plus"].get("residual_final_metrics", {}).get("MAE"),
        "e3_common_improvement_pct": e3["common_2022_plus"].get("improvement_percentage"),
        "e3_common_bootstrap": e3["common_2022_plus"].get("bootstrap"),
        "production_algorithm": production["selected_algorithm"],
        "e3_algorithm": e3["selected_algorithm"],
        "production_feature_count": production["feature_count"],
        "e3_feature_count": e3["feature_count"],
        "sanity_exact_match": None if sanity is None else sanity["exact_match"],
    }
    print("EXPERIMENT3_MULTIWINDOW_AUDIT=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
