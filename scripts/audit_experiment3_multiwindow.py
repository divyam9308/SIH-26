from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import BASELINE_FEATURES, CANDIDATE_FEATURES, as_of_feature_evidence, build_training_dataset, training_as_of_invariants
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, _select_regressor, temporal_project_split
from backend.app.ml.provenance import frame_fingerprint, git_commit_sha
from backend.app.ml.residual_overrun_experiment import (
    CURRENT_OVERRUN,
    FINAL_TARGET,
    RESIDUAL_TARGET,
    prepare_common_cost_cohort,
    reconstruct_final_overrun,
    run_residual_overrun_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audits"
STAGES = ("early", "mid", "late", "very_late")


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def feature_audit(train: pd.DataFrame) -> dict:
    return audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "same-snapshot late-stage signal; audited by production ablation",
            "cost_escalation_percentage": "same-snapshot current escalation; explicit Experiment 3 residual anchor",
        },
    )


def lifecycle_features(audit: dict) -> list[str]:
    return list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))


def stage_metrics(rows: pd.DataFrame, prediction_column: str) -> dict:
    out = {}
    for stage in STAGES:
        part = rows[rows.lifecycle_stage.eq(stage)] if "lifecycle_stage" in rows else rows.iloc[0:0]
        if part.empty:
            out[stage] = {"available": False}
            continue
        out[stage] = {
            "available": True,
            "MAE": _regression_metrics(
                part[FINAL_TARGET],
                part[prediction_column].to_numpy(dtype=float),
                part.sample_weight,
                part.canonical_project_id,
            )["MAE"],
            "rows": int(len(part)),
            "projects": int(part.canonical_project_id.nunique()),
        }
    return out


def weight_audit(frame: pd.DataFrame) -> dict:
    totals = frame.groupby("canonical_project_id")["sample_weight"].sum().astype(float)
    return {
        "min": round(float(totals.min()), 8),
        "median": round(float(totals.median()), 8),
        "max": round(float(totals.max()), 8),
        "all_close_to_one": bool(np.allclose(totals.to_numpy(), 1.0, atol=1e-8)),
    }


def project_diagnostics(rows: pd.DataFrame) -> dict:
    part = rows.copy()
    part["direct_abs"] = np.abs(part["direct_predicted_final_overrun"] - part[FINAL_TARGET])
    part["residual_abs"] = np.abs(part["residual_reconstructed_final_overrun"] - part[FINAL_TARGET])
    grouped = part.groupby("canonical_project_id")[["direct_abs", "residual_abs"]].mean()
    delta = grouped.direct_abs - grouped.residual_abs
    tie_tolerance = 0.1
    n = len(grouped)
    return {
        "projects": int(n),
        "residual_better_pct": round(float((delta > tie_tolerance).mean() * 100), 3),
        "direct_better_pct": round(float((delta < -tie_tolerance).mean() * 100), 3),
        "approx_tied_pct": round(float((delta.abs() <= tie_tolerance).mean() * 100), 3),
        "median_direct_abs_error": round(float(grouped.direct_abs.median()), 3),
        "median_residual_abs_error": round(float(grouped.residual_abs.median()), 3),
        "direct_p75_abs_error": round(float(grouped.direct_abs.quantile(0.75)), 3),
        "residual_p75_abs_error": round(float(grouped.residual_abs.quantile(0.75)), 3),
        "direct_p90_abs_error": round(float(grouped.direct_abs.quantile(0.90)), 3),
        "residual_p90_abs_error": round(float(grouped.residual_abs.quantile(0.90)), 3),
    }


def bootstrap_difference(rows: pd.DataFrame, samples: int = 1000, seed: int = 26103) -> dict:
    part = rows.copy()
    part["direct_abs"] = np.abs(part["direct_predicted_final_overrun"] - part[FINAL_TARGET])
    part["residual_abs"] = np.abs(part["residual_reconstructed_final_overrun"] - part[FINAL_TARGET])
    grouped = part.groupby("canonical_project_id")[["direct_abs", "residual_abs"]].mean()
    differences = (grouped.direct_abs - grouped.residual_abs).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    n = len(differences)
    for i in range(samples):
        sampled = differences[rng.integers(0, n, size=n)]
        draws[i] = sampled.mean()
    lower, upper = np.percentile(draws, [2.5, 97.5])
    point = float(differences.mean())
    if lower > 0:
        verdict = "residual_better"
    elif upper < 0:
        verdict = "residual_worse"
    else:
        verdict = "inconclusive"
    return {
        "metric": "project-balanced direct_MAE_minus_residual_MAE",
        "point_difference_pp": round(point, 3),
        "ci95_lower_pp": round(float(lower), 3),
        "ci95_upper_pp": round(float(upper), 3),
        "samples": int(samples),
        "verdict": verdict,
    }


def e3_evaluate(test: pd.DataFrame, features: list[str], direct_model, residual_model) -> tuple[dict, pd.DataFrame]:
    direct = np.asarray(direct_model.predict(test[features]), dtype=float)
    remaining = np.asarray(residual_model.predict(test[features]), dtype=float)
    reconstructed = reconstruct_final_overrun(test[CURRENT_OVERRUN], remaining)
    rows = test[[
        "canonical_project_id", "project_name", "snapshot_date", "completion_year", "lifecycle_stage",
        CURRENT_OVERRUN, FINAL_TARGET, RESIDUAL_TARGET, "sample_weight",
    ]].copy()
    rows["direct_predicted_final_overrun"] = direct
    rows["residual_predicted_remaining_overrun"] = remaining
    rows["residual_reconstructed_final_overrun"] = reconstructed
    direct_metrics = _regression_metrics(test[FINAL_TARGET], direct, test.sample_weight, test.canonical_project_id)
    residual_metrics = _regression_metrics(test[FINAL_TARGET], reconstructed, test.sample_weight, test.canonical_project_id)
    improvement = None
    if direct_metrics["MAE"] not in (None, 0) and residual_metrics["MAE"] is not None:
        improvement = round((direct_metrics["MAE"] - residual_metrics["MAE"]) / direct_metrics["MAE"] * 100, 3)
    return {
        "direct": direct_metrics,
        "residual_final": residual_metrics,
        "improvement_pct": improvement,
        "threshold_10pct_passed": bool(improvement is not None and improvement >= 10.0),
        "direct_stages": stage_metrics(rows, "direct_predicted_final_overrun"),
        "residual_stages": stage_metrics(rows, "residual_reconstructed_final_overrun"),
        "project_diagnostics": project_diagnostics(rows),
        "bootstrap": bootstrap_difference(rows),
    }, rows


def run(training_end: int) -> dict:
    training_start = 2001
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    years = data.completion_year.dropna().astype(int)
    latest = int(years.max())
    if training_end >= latest:
        raise ValueError("Training cutoff must precede latest completion year")

    dataset_info = {
        "source_commit": git_commit_sha(ROOT),
        "dataset_fingerprint": frame_fingerprint(data),
        "rows": int(len(data)),
        "projects": int(data.canonical_project_id.nunique()),
        "completion_year_min": int(years.min()),
        "completion_year_max": latest,
        "identity_rows": int(len(identity)),
        "identity_verified_rows": int(identity.identity_verified.sum()),
    }

    # Production lifecycle cost path: same feature audit, seed and selector as monthly_training._train_variant.
    prod_train, prod_test = temporal_project_split(data, training_start, training_end, latest)
    prod_train_inv = training_as_of_invariants(prod_train)
    prod_test_inv = training_as_of_invariants(prod_test)
    prod_audit = feature_audit(prod_train)
    prod_features = lifecycle_features(prod_audit)
    prod_algorithm, prod_model, prod_comparison = _select_regressor(prod_train, prod_features, FINAL_TARGET, 26203)
    prod_pred = np.asarray(prod_model.predict(prod_test[prod_features]), dtype=float)
    prod_metrics = _regression_metrics(prod_test[FINAL_TARGET], prod_pred, prod_test.sample_weight, prod_test.canonical_project_id)
    prod_rows = prod_test[["canonical_project_id", "lifecycle_stage", FINAL_TARGET, "sample_weight"]].copy()
    prod_rows["prediction"] = prod_pred

    common_prod_test = prod_test[prod_test.completion_year.between(2022, latest)].copy()
    common_prod = None
    if not common_prod_test.empty:
        pred = np.asarray(prod_model.predict(common_prod_test[prod_features]), dtype=float)
        common_prod = _regression_metrics(common_prod_test[FINAL_TARGET], pred, common_prod_test.sample_weight, common_prod_test.canonical_project_id)

    # Experiment 3 exact controlled cohort and exact current algorithm path.
    e3_train, e3_test = prepare_common_cost_cohort(data, training_start, training_end, latest)
    e3_audit = feature_audit(e3_train)
    e3_features = lifecycle_features(e3_audit)
    e3_algorithm, e3_direct_model, e3_internal = _select_regressor(e3_train, e3_features, FINAL_TARGET, 27103)
    e3_residual_model = _fit_pipeline(_regressors(27103)[e3_algorithm], e3_train, e3_features, RESIDUAL_TARGET)
    natural_e3, natural_rows = e3_evaluate(e3_test, e3_features, e3_direct_model, e3_residual_model)

    common_e3_test = e3_test[e3_test.completion_year.between(2022, latest)].copy()
    common_e3 = None
    if not common_e3_test.empty:
        common_e3, _ = e3_evaluate(common_e3_test, e3_features, e3_direct_model, e3_residual_model)

    sanity = None
    if training_end == 2015:
        official = run_residual_overrun_experiment(2001, 2015, data=data, persist=False)
        sanity = {
            "official_direct_mae": official["direct_final_overrun_metrics"]["MAE"],
            "official_residual_mae": official["residual_reconstructed_final_overrun_metrics"]["MAE"],
            "official_improvement_pct": official["final_mae_improvement_percentage"],
            "runner_direct_mae": natural_e3["direct"]["MAE"],
            "runner_residual_mae": natural_e3["residual_final"]["MAE"],
            "runner_improvement_pct": natural_e3["improvement_pct"],
        }
        sanity["matches"] = bool(
            abs(sanity["official_direct_mae"] - sanity["runner_direct_mae"]) < 1e-9
            and abs(sanity["official_residual_mae"] - sanity["runner_residual_mae"]) < 1e-9
        )

    stage_compare = {}
    for stage in STAGES:
        direct = natural_e3["direct_stages"].get(stage, {})
        residual = natural_e3["residual_stages"].get(stage, {})
        d = direct.get("MAE")
        r = residual.get("MAE")
        imp = round((d - r) / d * 100, 3) if d not in (None, 0) and r is not None else None
        stage_compare[stage] = {
            "direct_mae": d,
            "residual_mae": r,
            "improvement_pct": imp,
            "rows": direct.get("rows"),
            "projects": direct.get("projects"),
        }

    result = {
        "audit": "experiment3_multiwindow_cost_only",
        "training_period": [training_start, training_end],
        "natural_holdout": [training_end + 1, latest],
        "common_holdout": [2022, latest],
        "dataset": dataset_info,
        "production": {
            "train_rows": int(len(prod_train)),
            "train_projects": int(prod_train.canonical_project_id.nunique()),
            "test_rows": int(len(prod_test)),
            "test_projects": int(prod_test.canonical_project_id.nunique()),
            "algorithm": prod_algorithm,
            "features": prod_features,
            "feature_count": len(prod_features),
            "feature_quality_score": prod_audit["data_quality_score"],
            "as_of_evidence_coverage": prod_audit["as_of_evidence_coverage"],
            "removed_features": prod_audit["removed_features"],
            "internal_algorithm_comparison": prod_comparison,
            "train_as_of_invariants": prod_train_inv,
            "test_as_of_invariants": prod_test_inv,
            "natural_metrics": prod_metrics,
            "natural_stage_metrics": stage_metrics(prod_rows, "prediction"),
            "common_2022_plus_metrics": common_prod,
        },
        "experiment3": {
            "train_rows": int(len(e3_train)),
            "train_projects": int(e3_train.canonical_project_id.nunique()),
            "test_rows": int(len(e3_test)),
            "test_projects": int(e3_test.canonical_project_id.nunique()),
            "algorithm": e3_algorithm,
            "features": e3_features,
            "feature_count": len(e3_features),
            "feature_quality_score": e3_audit["data_quality_score"],
            "as_of_evidence_coverage": e3_audit["as_of_evidence_coverage"],
            "removed_features": e3_audit["removed_features"],
            "internal_algorithm_comparison": e3_internal,
            "train_weight_audit": weight_audit(e3_train),
            "test_weight_audit": weight_audit(e3_test),
            "comparison_control": {
                "same_train_snapshots": True,
                "same_test_snapshots": True,
                "same_features": True,
                "same_regressor_family": True,
                "post_sampling_project_weights": True,
                "common_cohort_weights_renormalized": True,
                "explicit_as_of_lineage": e3_audit["as_of_evidence_coverage"] == 100.0,
            },
            "natural": natural_e3,
            "common_2022_plus": common_e3,
            "stage_comparison": stage_compare,
            "sanity_check_2001_2015": sanity,
        },
        "cohort_warning": "Production full-cohort MAE and Experiment 3 MAEs are not directly comparable because Experiment 3 requires non-null current cost escalation. E3 direct-vs-residual is the controlled target-formulation comparison.",
    }
    return safe(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", type=int, required=True)
    args = parser.parse_args()
    result = run(args.training_end)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"experiment3_cost_audit_2001_{args.training_end}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    e3 = result["experiment3"]["natural"]
    common = result["experiment3"]["common_2022_plus"]
    summary = {
        "training_end": args.training_end,
        "natural_holdout": result["natural_holdout"],
        "production_mae": result["production"]["natural_metrics"]["MAE"],
        "production_common_mae": (result["production"]["common_2022_plus_metrics"] or {}).get("MAE"),
        "e3_direct_mae": e3["direct"]["MAE"],
        "e3_residual_mae": e3["residual_final"]["MAE"],
        "e3_improvement_pct": e3["improvement_pct"],
        "e3_threshold_passed": e3["threshold_10pct_passed"],
        "e3_common_direct_mae": (common or {}).get("direct", {}).get("MAE"),
        "e3_common_residual_mae": (common or {}).get("residual_final", {}).get("MAE"),
        "e3_common_improvement_pct": (common or {}).get("improvement_pct"),
        "algorithm_production": result["production"]["algorithm"],
        "algorithm_e3": result["experiment3"]["algorithm"],
        "feature_count_production": result["production"]["feature_count"],
        "feature_count_e3": result["experiment3"]["feature_count"],
        "train_projects_e3": result["experiment3"]["train_projects"],
        "test_projects_e3": result["experiment3"]["test_projects"],
        "weights_train_ok": result["experiment3"]["train_weight_audit"]["all_close_to_one"],
        "weights_test_ok": result["experiment3"]["test_weight_audit"]["all_close_to_one"],
        "bootstrap": e3["bootstrap"],
        "projects": e3["project_diagnostics"],
        "stages": result["experiment3"]["stage_comparison"],
        "sanity": result["experiment3"]["sanity_check_2001_2015"],
        "dataset": result["dataset"],
    }
    print("AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True, allow_nan=False))
    print(f"AUDIT_FILE={path}")


if __name__ == "__main__":
    main()
