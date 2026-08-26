"""Controlled current-production-vs-Experiment-13 audit.

The runner reproduces the current production training contract far enough to
compare cost and delay without risk/SHAP publication overhead:

1. select production lifecycle features and algorithms on the raw training data;
2. reproduce the promoted Exp12 cost trajectory feature selection and refit;
3. retain the ordinary production delay model;
4. fit Exp13 through its registered adapter on the same frozen evidence;
5. write a JSON evidence artifact for one temporal window.

This is an audit utility. It does not promote or mutate production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.adapter_exp13 import fit_against_production
from backend.app.ml.experiments.trajectory_exp12 import _safe
from backend.app.ml.experiments.trajectory_exp12_v2 import (
    _select_target_features as select_exp12_cost_features,
    _usable_features as usable_exp12_features,
)
from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
)
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regressors,
    _select_regressor,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_BASELINE,
    PRODUCTION_COST_SEED,
    enrich_supervised_for_production,
)


def current_production_cost_delay(
    data: pd.DataFrame,
    start: int,
    end: int,
    test_end: int,
) -> tuple[dict, dict]:
    raw_train, _ = temporal_project_split(data, start, end, test_end)
    audit = audit_features(
        raw_train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by production ablation",
            "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by production ablation",
        },
    )
    base_features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))

    cost_name, _raw_cost, cost_comparisons = _select_regressor(
        raw_train,
        base_features,
        "actual_cost_overrun_percentage",
        PRODUCTION_COST_SEED,
    )
    delay_name, delay_model, delay_comparisons = _select_regressor(
        raw_train,
        base_features,
        "actual_delay_days",
        26204,
    )

    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(
        enriched.completion_year, errors="coerce"
    )
    exp12_train, _ = temporal_project_split(enriched, start, end, test_end)
    usable, trajectory_audit = usable_exp12_features(exp12_train)
    cost_added, cost_group, feature_comparisons = select_exp12_cost_features(
        exp12_train,
        base_features,
        usable,
        "actual_cost_overrun_percentage",
        cost_name,
        PRODUCTION_COST_SEED,
    )
    cost_features = list(dict.fromkeys(base_features + cost_added))
    cost_model = _fit_pipeline(
        _regressors(PRODUCTION_COST_SEED)[cost_name],
        exp12_train,
        cost_features,
        "actual_cost_overrun_percentage",
    )

    metadata = {
        "features_used": base_features,
        "cost_features_used": cost_features,
        "delay_features_used": base_features,
        "risk_features_used": base_features,
        "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "promoted_from_experiment": "exp_12",
        "cost_trajectory_feature_group": cost_group,
        "cost_trajectory_features": cost_added,
        "trajectory_feature_availability": trajectory_audit,
        "internal_cost_trajectory_feature_comparisons": feature_comparisons,
    }
    bundle = {
        "metadata": metadata,
        "cost": cost_model,
        "delay": delay_model,
    }
    receipt = {
        "run_id": f"exp13-audit-production-{start}-{end}",
        "dataset_fingerprint": "ephemeral-controlled-audit",
        "features_used": base_features,
        "cost_features_used": cost_features,
        "delay_features_used": base_features,
        "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "production_algorithm_comparisons": {
            "cost": cost_comparisons,
            "delay": delay_comparisons,
        },
        "production_exp12_cost_feature_group": cost_group,
        "production_exp12_cost_features": cost_added,
    }
    return bundle, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    test_end = int(data.completion_year.dropna().max())
    bundle, receipt = current_production_cost_delay(
        data,
        args.start,
        args.end,
        test_end,
    )
    fitted = fit_against_production(
        data=data,
        training_start=args.start,
        training_end=args.end,
        test_end=test_end,
        production_bundle=bundle,
        production_receipt=receipt,
    )

    overall = fitted["overall_comparison"]
    experiment = fitted["experiment"]
    payload = {
        "window": f"{args.start}_{args.end}",
        "test_end": test_end,
        "audit_mode": (
            "current promoted Exp12 production cost contract plus current production delay; "
            "risk/SHAP/publication skipped"
        ),
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "production_selected_algorithms": receipt["selected_algorithms"],
        "production_exp12_cost_feature_group": receipt[
            "production_exp12_cost_feature_group"
        ],
        "production_exp12_cost_features": receipt["production_exp12_cost_features"],
        "experiment": experiment,
        "overall_comparison": overall,
    }
    safe_payload = _safe(payload)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_payload, indent=2, allow_nan=False) + "\n")

    summary = {
        "window": payload["window"],
        "test_end": test_end,
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "production_cost_mae": overall.get("production_cost_mae"),
        "experiment_cost_mae": overall.get("experiment_cost_mae"),
        "cost_improvement_percentage": overall.get("improvement_percentage"),
        "production_delay_mae": overall.get("production_delay_mae"),
        "experiment_delay_mae": overall.get("experiment_delay_mae"),
        "delay_improvement_percentage": overall.get(
            "delay_improvement_percentage"
        ),
        "comparison_test_projects": overall.get("comparison_test_projects"),
        "comparison_test_snapshots": overall.get("comparison_test_snapshots"),
        "paired_project_cost_comparison": overall.get(
            "paired_project_cost_comparison"
        ),
        "paired_project_delay_comparison": overall.get(
            "paired_project_delay_comparison"
        ),
        "stage_balanced": overall.get("stage_balanced"),
        "internal_feature_selection": overall.get("internal_feature_selection"),
        "selected_feature_groups": experiment.get("selected_feature_groups"),
        "cost_added_features": experiment.get("cost_added_features"),
        "delay_added_features": experiment.get("delay_added_features"),
    }
    print(
        "EXP13_FAST_COMPARISON="
        + json.dumps(_safe(summary), sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
