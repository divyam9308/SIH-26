"""Fast exact Cost/Delay audit against the current promoted production contract.

This preserves production's lifecycle feature audit, temporal model selection,
Exp12 cost trajectory selection, seeds, weights and holdout. It intentionally
skips risk fitting, production ablations, SHAP and artifact publication because
those cannot change the Cost/Delay predictions being compared here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.trajectory_exp12_v2 import _select_target_features, _usable_features
from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import BASELINE_FEATURES, CANDIDATE_FEATURES, as_of_feature_evidence, build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, _select_regressor, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_BASELINE, PRODUCTION_COST_SEED, enrich_supervised_for_production


def fast_current_production(data: pd.DataFrame, start: int, end: int, test_end: int) -> tuple[dict, dict]:
    train, _ = temporal_project_split(data, start, end, test_end)
    audit = audit_features(
        train, CANDIDATE_FEATURES, minimum_availability=10, minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by production ablation",
            "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by production ablation",
        },
    )
    production_features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    cost_name, _discarded_cost, cost_cmp = _select_regressor(train, production_features, "actual_cost_overrun_percentage", 26203)
    delay_name, delay_model, delay_cmp = _select_regressor(train, production_features, "actual_delay_days", 26204)
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    enriched_train, _ = temporal_project_split(enriched, start, end, test_end)
    usable, trajectory_audit = _usable_features(enriched_train)
    cost_added, cost_group, cost_feature_cmp = _select_target_features(
        enriched_train, production_features, usable, "actual_cost_overrun_percentage", cost_name, PRODUCTION_COST_SEED
    )
    cost_features = list(dict.fromkeys(production_features + cost_added))
    cost_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[cost_name], enriched_train, cost_features, "actual_cost_overrun_percentage")
    selected = {"cost": cost_name, "delay": delay_name}
    metadata = {
        "features_used": production_features, "cost_features_used": cost_features, "delay_features_used": production_features,
        "selected_algorithms": selected, "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "cost_trajectory_feature_group": cost_group, "cost_trajectory_features": cost_added,
        "trajectory_feature_availability": trajectory_audit, "internal_cost_trajectory_feature_comparisons": cost_feature_cmp,
    }
    bundle = {"metadata": metadata, "cost": cost_model, "delay": delay_model}
    receipt = {
        "run_id": f"fast-current-production-{start}-{end}", "selected_algorithms": selected,
        "features_used": production_features, "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "internal_algorithm_comparisons": {"cost": cost_cmp, "delay": delay_cmp},
    }
    return bundle, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--test-end", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, _identity = build_training_dataset()
    data = data.copy(); data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    bundle, receipt = fast_current_production(data, args.start, args.end, args.test_end)
    adapter = get_experiment_adapter(args.experiment)
    fitted = adapter.module.fit_against_production(
        data=data, training_start=args.start, training_end=args.end, test_end=args.test_end,
        production_bundle=bundle, production_receipt=receipt,
    )
    overall = dict(fitted.get("overall_comparison") or {}); experiment = dict(fitted.get("experiment") or {})
    payload = {
        "window": f"{args.start}_{args.end}", "test_end": args.test_end,
        "audit_mode": "exact current production Cost/Delay training; skips risk, SHAP, ablations and publication only",
        "production": receipt, "production_metadata": bundle["metadata"], "experiment": experiment, "overall_comparison": overall,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")
    summary = {
        "window": payload["window"], "experiment_id": args.experiment,
        "production_cost_mae": overall.get("production_cost_mae"), "experiment_cost_mae": overall.get("experiment_cost_mae"),
        "cost_improvement_percentage": overall.get("cost_improvement_percentage", overall.get("improvement_percentage")),
        "production_delay_mae": overall.get("production_delay_mae"), "experiment_delay_mae": overall.get("experiment_delay_mae"),
        "delay_improvement_percentage": overall.get("delay_improvement_percentage"),
        "comparison_test_projects": overall.get("comparison_test_projects"), "comparison_test_snapshots": overall.get("comparison_test_snapshots"),
        "decision": overall.get("decision", experiment.get("decision")),
    }
    print("FAST_CURRENT_COMPARISON=" + json.dumps(summary, sort_keys=True, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
