from __future__ import annotations

import argparse
import json

import pandas as pd

from backend.app.ml.monthly_lifecycle import training_as_of_invariants
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, _select_regressor, temporal_project_split
from scripts.audit_snapshot_density_cost import (
    DENSITIES,
    OUT,
    ROOT,
    TARGET,
    density_sample,
    feature_audit,
    fixed_eval_metrics,
    lifecycle_features,
    prepare,
    safe,
    weight_audit,
)


def run(training_end: int) -> dict:
    training_start = 2001
    engineered, all_pool, cost_pool, funnel, _ = prepare()

    # Earlier audit runs showed these cohorts have identical counts. For this
    # optimized runner, prove they are the exact same engineered rows before
    # skipping duplicate cost-only model fits.
    all_index = all_pool.index.sort_values()
    cost_index = cost_pool.index.sort_values()
    cohorts_identical = bool(len(all_index) == len(cost_index) and all_index.equals(cost_index))
    if not cohorts_identical:
        raise AssertionError(
            "Cannot skip cost-only duplicate fits: all-target and cost-only row sets differ"
        )

    years = pd.to_numeric(engineered.completion_year, errors="coerce").dropna().astype(int)
    latest = int(years.max())
    sampled = {density: density_sample(all_pool, density) for density in DENSITIES}

    baseline_data = sampled["quarterly"]
    baseline_train, baseline_test = temporal_project_split(
        baseline_data, training_start, training_end, latest
    )
    audit = feature_audit(baseline_train)
    features = lifecycle_features(audit)
    algorithm, baseline_model, algorithm_comparison = _select_regressor(
        baseline_train, features, TARGET, 26203
    )
    baseline_fixed = fixed_eval_metrics(baseline_model, baseline_test, features)
    common_test = baseline_test[
        pd.to_numeric(baseline_test.completion_year, errors="coerce").between(2022, latest)
    ].copy()
    baseline_common = fixed_eval_metrics(baseline_model, common_test, features) if len(common_test) else None

    variants = {}
    for density in DENSITIES:
        train, native_test = temporal_project_split(
            sampled[density], training_start, training_end, latest
        )
        invariants = training_as_of_invariants(train)
        if not invariants["passed"]:
            raise ValueError(f"As-of invariant failed for {density}: {invariants}")

        if density == "quarterly":
            model = baseline_model
        else:
            model = _fit_pipeline(_regressors(26203)[algorithm], train, features, TARGET)

        fixed = fixed_eval_metrics(model, baseline_test, features)
        common = fixed_eval_metrics(model, common_test, features) if len(common_test) else None
        base_mae = baseline_fixed["metrics"]["MAE"]
        variant_mae = fixed["metrics"]["MAE"]
        improvement = round((base_mae - variant_mae) / base_mae * 100, 3) if base_mae else None
        common_improvement = None
        if baseline_common and common:
            b = baseline_common["metrics"]["MAE"]
            v = common["metrics"]["MAE"]
            common_improvement = round((b - v) / b * 100, 3) if b else None

        variants[density] = {
            "training_rows": int(len(train)),
            "training_projects": int(train.canonical_project_id.nunique()),
            "native_test_rows": int(len(native_test)),
            "native_test_projects": int(native_test.canonical_project_id.nunique()),
            "training_weight_audit": weight_audit(train),
            "fixed_quarterly_eval": fixed,
            "fixed_eval_improvement_vs_current_quarterly_pct": improvement,
            "common_2022_plus_fixed_eval": common,
            "common_improvement_vs_current_quarterly_pct": common_improvement,
        }

    return safe({
        "audit": "snapshot_density_cost_only_optimized_identical_cohorts",
        "training_period": [training_start, training_end],
        "holdout_period": [training_end + 1, latest],
        "common_holdout": [2022, latest],
        "funnel": funnel,
        "control": {
            "all_target_and_cost_only_row_sets_identical": cohorts_identical,
            "same_fixed_test_snapshots_across_variants": True,
            "same_features_across_variants": True,
            "same_regressor_family_across_variants": True,
            "same_hyperparameter_constructor_across_variants": True,
            "same_temporal_cutoff_across_variants": True,
            "project_balanced_weights_after_sampling": True,
            "locked_algorithm": algorithm,
            "feature_count": len(features),
            "features": features,
            "baseline_algorithm_comparison": algorithm_comparison,
        },
        "baseline": {
            "training_rows": int(len(baseline_train)),
            "training_projects": int(baseline_train.canonical_project_id.nunique()),
            "fixed_quarterly_eval": baseline_fixed,
            "common_2022_plus_eval": baseline_common,
        },
        "variants": variants,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", type=int, required=True)
    args = parser.parse_args()
    result = run(args.training_end)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"snapshot_density_cost_fast_audit_2001_{args.training_end}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    compact = {
        "training_end": args.training_end,
        "cohorts_identical": result["control"]["all_target_and_cost_only_row_sets_identical"],
        "algorithm": result["control"]["locked_algorithm"],
        "baseline_mae": result["baseline"]["fixed_quarterly_eval"]["metrics"]["MAE"],
        "variants": {
            density: {
                "train_rows": result["variants"][density]["training_rows"],
                "train_projects": result["variants"][density]["training_projects"],
                "mae": result["variants"][density]["fixed_quarterly_eval"]["metrics"]["MAE"],
                "improvement_pct": result["variants"][density]["fixed_eval_improvement_vs_current_quarterly_pct"],
                "common_mae": result["variants"][density]["common_2022_plus_fixed_eval"]["metrics"]["MAE"] if result["variants"][density]["common_2022_plus_fixed_eval"] else None,
                "common_improvement_pct": result["variants"][density]["common_improvement_vs_current_quarterly_pct"],
            }
            for density in DENSITIES
        },
    }
    print("SNAPSHOT_DENSITY_FAST_AUDIT=" + json.dumps(compact, sort_keys=True))
    print("AUDIT_FILE=" + str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
