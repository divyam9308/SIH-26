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
    OUTCOMES,
    TARGETS,
    as_of_feature_evidence,
    assign_project_balanced_weights,
    engineer_as_of_features,
    load_monthly_snapshots,
    resolve_identities,
    training_as_of_invariants,
)
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    _select_regressor,
    temporal_project_split,
)
from backend.app.ml.provenance import frame_fingerprint, git_commit_sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audits"
TARGET = "actual_cost_overrun_percentage"
DENSITIES = ("quarterly", "bimonthly", "monthly")
COHORTS = ("all_targets", "cost_only")
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
            "cost_escalation_percentage": "same-snapshot current escalation",
        },
    )


def lifecycle_features(audit: dict) -> list[str]:
    return list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))


def density_sample(frame: pd.DataFrame, density: str) -> pd.DataFrame:
    data = frame.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    if density == "quarterly":
        data["_density_bucket"] = data.snapshot_date.dt.to_period("Q").astype(str)
    elif density == "bimonthly":
        year = data.snapshot_date.dt.year.astype("Int64").astype(str)
        bucket = ((data.snapshot_date.dt.month - 1) // 2 + 1).astype("Int64").astype(str)
        data["_density_bucket"] = year + "-B" + bucket
    elif density == "monthly":
        data["_density_bucket"] = data.snapshot_date.dt.to_period("M").astype(str)
    else:
        raise ValueError(f"Unknown density: {density}")
    sampled = (
        data.sort_values("snapshot_date")
        .drop_duplicates(["canonical_project_id", "_density_bucket"], keep="last")
        .drop(columns=["_density_bucket"])
    )
    return assign_project_balanced_weights(sampled)


def weight_audit(frame: pd.DataFrame) -> dict:
    totals = frame.groupby("canonical_project_id")["sample_weight"].sum().astype(float)
    return {
        "min": round(float(totals.min()), 10) if len(totals) else None,
        "median": round(float(totals.median()), 10) if len(totals) else None,
        "max": round(float(totals.max()), 10) if len(totals) else None,
        "all_close_to_one": bool(len(totals) and np.allclose(totals.to_numpy(), 1.0, atol=1e-10)),
    }


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return _regression_metrics(
        frame[TARGET], prediction, frame.sample_weight, frame.canonical_project_id
    )


def stage_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    rows = frame[["canonical_project_id", "lifecycle_stage", TARGET, "sample_weight"]].copy()
    rows["prediction"] = prediction
    result = {}
    for stage in STAGES:
        part = rows[rows.lifecycle_stage.eq(stage)]
        result[stage] = (
            {"available": False}
            if part.empty
            else {
                "available": True,
                "rows": int(len(part)),
                "projects": int(part.canonical_project_id.nunique()),
                "MAE": _regression_metrics(
                    part[TARGET], part.prediction, part.sample_weight, part.canonical_project_id
                )["MAE"],
            }
        )
    return result


def fixed_eval_metrics(model, frame: pd.DataFrame, features: list[str]) -> dict:
    prediction = np.asarray(model.predict(frame[features]), dtype=float)
    return {
        "metrics": metrics(frame, prediction),
        "stages": stage_metrics(frame, prediction),
        "rows": int(len(frame)),
        "projects": int(frame.canonical_project_id.nunique()),
    }


def project_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["canonical_project_id", "project_name", "completion_year", "eligible_rows"])
    work = frame.copy()
    work["completion_year"] = pd.to_numeric(work["completion_year"], errors="coerce")
    grouped = work.groupby("canonical_project_id", dropna=False)
    rows = []
    for project_id, part in grouped:
        names = part.project_name.dropna().astype(str)
        years = part.completion_year.dropna().astype(int)
        rows.append(
            {
                "canonical_project_id": project_id,
                "project_name": names.iloc[-1] if len(names) else None,
                "completion_year": int(years.iloc[0]) if len(years) else None,
                "eligible_rows": int(len(part)),
                "all_delay_targets_missing": bool(part.actual_delay_days.isna().all()),
                "all_risk_targets_missing": bool(part.actual_risk.isna().all()),
            }
        )
    return pd.DataFrame(rows)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    snapshots = load_monthly_snapshots()
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, identity = resolve_identities(snapshots, outcomes)
    engineered = engineer_as_of_features(resolved, outcomes)
    engineered["snapshot_date"] = pd.to_datetime(engineered["snapshot_date"], errors="coerce")
    engineered["completion_date"] = pd.to_datetime(engineered["completion_date"], errors="coerce")

    verified = engineered.identity_verified.eq(True)
    pre_completion = engineered.snapshot_date.lt(engineered.completion_date)
    cost_target = engineered[TARGET].notna()
    all_targets = engineered[TARGETS].notna().all(axis=1)

    cost_pool = engineered[verified & cost_target & pre_completion].copy()
    all_pool = engineered[verified & all_targets & pre_completion].copy()

    current_quarterly = density_sample(all_pool, "quarterly")
    current_projects = set(current_quarterly.canonical_project_id.astype(str))
    cost_projects = set(cost_pool.canonical_project_id.astype(str))
    extra_ids = cost_projects - current_projects
    extras = project_summary(cost_pool[cost_pool.canonical_project_id.astype(str).isin(extra_ids)])

    identity_counts = (
        identity.groupby("identity_method").agg(rows=("row_index", "size"), projects=("canonical_project_id", "nunique")).reset_index()
    )

    funnel = {
        "raw_monthly_rows": int(len(snapshots)),
        "resolved_rows": int(len(resolved)),
        "resolved_canonical_projects": int(resolved.canonical_project_id.nunique()),
        "identity_verified_rows": int(verified.sum()),
        "identity_verified_projects": int(engineered.loc[verified, "canonical_project_id"].nunique()),
        "verified_cost_target_rows": int((verified & cost_target).sum()),
        "verified_cost_target_projects": int(engineered.loc[verified & cost_target, "canonical_project_id"].nunique()),
        "cost_only_precompletion_rows": int(len(cost_pool)),
        "cost_only_precompletion_projects": int(cost_pool.canonical_project_id.nunique()),
        "current_all_targets_precompletion_rows": int(len(all_pool)),
        "current_all_targets_precompletion_projects": int(all_pool.canonical_project_id.nunique()),
        "current_quarterly_rows": int(len(current_quarterly)),
        "current_quarterly_projects": int(current_quarterly.canonical_project_id.nunique()),
        "additional_cost_only_projects": int(len(extra_ids)),
        "additional_cost_only_rows_before_sampling": int(len(cost_pool[cost_pool.canonical_project_id.astype(str).isin(extra_ids)])),
        "identity_method_counts": identity_counts.to_dict(orient="records"),
        "density_counts": {},
    }
    for cohort_name, pool in (("all_targets", all_pool), ("cost_only", cost_pool)):
        funnel["density_counts"][cohort_name] = {}
        for density in DENSITIES:
            sampled = density_sample(pool, density)
            funnel["density_counts"][cohort_name][density] = {
                "rows": int(len(sampled)),
                "projects": int(sampled.canonical_project_id.nunique()),
                "weight_audit": weight_audit(sampled),
            }

    return engineered, all_pool, cost_pool, funnel, extras


def run(training_end: int) -> dict:
    training_start = 2001
    engineered, all_pool, cost_pool, funnel, extras = prepare()
    years = pd.to_numeric(engineered.completion_year, errors="coerce").dropna().astype(int)
    latest = int(years.max())
    if training_end >= latest:
        raise ValueError("Training end must precede latest completion year")

    sampled = {
        cohort: {
            density: density_sample(pool, density)
            for density in DENSITIES
        }
        for cohort, pool in (("all_targets", all_pool), ("cost_only", cost_pool))
    }

    # Baseline = current production eligibility + current quarterly sampling.
    baseline_data = sampled["all_targets"]["quarterly"]
    baseline_train, baseline_test = temporal_project_split(
        baseline_data, training_start, training_end, latest
    )
    baseline_audit = feature_audit(baseline_train)
    features = lifecycle_features(baseline_audit)
    algorithm, baseline_model, algorithm_comparison = _select_regressor(
        baseline_train, features, TARGET, 26203
    )
    baseline_fixed = fixed_eval_metrics(baseline_model, baseline_test, features)
    common_fixed_test = baseline_test[
        pd.to_numeric(baseline_test.completion_year, errors="coerce").between(2022, latest)
    ].copy()
    baseline_common = fixed_eval_metrics(baseline_model, common_fixed_test, features) if len(common_fixed_test) else None

    variants = {}
    for cohort in COHORTS:
        variants[cohort] = {}
        for density in DENSITIES:
            data = sampled[cohort][density]
            train, native_test = temporal_project_split(data, training_start, training_end, latest)
            invariants = training_as_of_invariants(train)
            if not invariants["passed"]:
                raise ValueError(f"As-of invariant failed for {cohort}/{density}: {invariants}")
            model = _fit_pipeline(_regressors(26203)[algorithm], train, features, TARGET)

            # Primary comparison: every variant is evaluated on the exact same
            # current-production quarterly test snapshots.
            fixed = fixed_eval_metrics(model, baseline_test, features)
            common = fixed_eval_metrics(model, common_fixed_test, features) if len(common_fixed_test) else None

            # Coverage diagnostic only: cost-only variants can also be evaluated
            # on their larger native quarterly cost-only test population.
            expanded = None
            if cohort == "cost_only":
                _, expanded_test = temporal_project_split(
                    sampled["cost_only"]["quarterly"], training_start, training_end, latest
                )
                expanded = fixed_eval_metrics(model, expanded_test, features)

            base_mae = baseline_fixed["metrics"]["MAE"]
            variant_mae = fixed["metrics"]["MAE"]
            improvement = round((base_mae - variant_mae) / base_mae * 100, 3) if base_mae else None
            common_improvement = None
            if baseline_common and common:
                b = baseline_common["metrics"]["MAE"]
                v = common["metrics"]["MAE"]
                common_improvement = round((b - v) / b * 100, 3) if b else None

            variants[cohort][density] = {
                "training_rows": int(len(train)),
                "training_projects": int(train.canonical_project_id.nunique()),
                "native_test_rows": int(len(native_test)),
                "native_test_projects": int(native_test.canonical_project_id.nunique()),
                "training_weight_audit": weight_audit(train),
                "fixed_quarterly_eval": fixed,
                "fixed_eval_improvement_vs_current_quarterly_pct": improvement,
                "common_2022_plus_fixed_eval": common,
                "common_improvement_vs_current_quarterly_pct": common_improvement,
                "expanded_cost_only_quarterly_eval": expanded,
            }

    # Persist full additional-project inventory as audit artifact only.
    OUT.mkdir(parents=True, exist_ok=True)
    extras_path = OUT / f"snapshot_density_additional_cost_projects_2001_{training_end}.csv"
    extras.to_csv(extras_path, index=False)

    result = {
        "audit": "snapshot_density_cost_only",
        "source_commit": git_commit_sha(ROOT),
        "training_period": [training_start, training_end],
        "holdout_period": [training_end + 1, latest],
        "common_holdout": [2022, latest],
        "dataset_fingerprint_engineered": frame_fingerprint(engineered),
        "funnel": funnel,
        "control": {
            "primary_evaluation_snapshot_set": "current all-target quarterly test snapshots",
            "same_fixed_test_snapshots_across_variants": True,
            "same_features_across_variants": True,
            "same_regressor_family_across_variants": True,
            "same_hyperparameter_constructor_across_variants": True,
            "same_temporal_cutoff_across_variants": True,
            "project_balanced_weights_after_sampling": True,
            "feature_count": int(len(features)),
            "features": features,
            "locked_algorithm": algorithm,
            "baseline_feature_quality_score": baseline_audit["data_quality_score"],
            "baseline_as_of_evidence_coverage": baseline_audit["as_of_evidence_coverage"],
            "baseline_algorithm_comparison": algorithm_comparison,
        },
        "baseline": {
            "training_rows": int(len(baseline_train)),
            "training_projects": int(baseline_train.canonical_project_id.nunique()),
            "test_rows": int(len(baseline_test)),
            "test_projects": int(baseline_test.canonical_project_id.nunique()),
            "fixed_quarterly_eval": baseline_fixed,
            "common_2022_plus_eval": baseline_common,
        },
        "variants": variants,
        "additional_project_inventory_file": str(extras_path.relative_to(ROOT)),
        "interpretation_rule": "Positive improvement means lower MAE than the current all-target quarterly baseline on the exact same fixed quarterly test snapshots.",
    }
    return safe(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", type=int, required=True)
    args = parser.parse_args()
    result = run(args.training_end)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"snapshot_density_cost_audit_2001_{args.training_end}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))

    compact = {
        "training_end": args.training_end,
        "raw_rows": result["funnel"]["raw_monthly_rows"],
        "current_rows": result["funnel"]["current_quarterly_rows"],
        "current_projects": result["funnel"]["current_quarterly_projects"],
        "cost_only_pool_rows": result["funnel"]["cost_only_precompletion_rows"],
        "cost_only_pool_projects": result["funnel"]["cost_only_precompletion_projects"],
        "additional_cost_only_projects": result["funnel"]["additional_cost_only_projects"],
        "algorithm": result["control"]["locked_algorithm"],
        "baseline_mae": result["baseline"]["fixed_quarterly_eval"]["metrics"]["MAE"],
        "variants": {
            f"{cohort}:{density}": {
                "train_rows": result["variants"][cohort][density]["training_rows"],
                "train_projects": result["variants"][cohort][density]["training_projects"],
                "mae": result["variants"][cohort][density]["fixed_quarterly_eval"]["metrics"]["MAE"],
                "improvement_pct": result["variants"][cohort][density]["fixed_eval_improvement_vs_current_quarterly_pct"],
            }
            for cohort in COHORTS for density in DENSITIES
        },
    }
    print("SNAPSHOT_DENSITY_AUDIT=" + json.dumps(compact, sort_keys=True))
    print("AUDIT_FILE=" + str(path))
    print("EXTRA_PROJECTS_FILE=" + result["additional_project_inventory_file"])


if __name__ == "__main__":
    main()
