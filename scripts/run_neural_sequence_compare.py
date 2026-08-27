#!/usr/bin/env python3
"""Controlled production-vs-neural-sequence comparison for isolated PRs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.trajectory_exp12_v2 import _select_target_features as select_exp12_features
from backend.app.ml.experiments.trajectory_exp12_v2 import _usable_features as usable_exp12_features
from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import BASELINE_FEATURES, CANDIDATE_FEATURES, as_of_feature_evidence, build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _json_safe, _regressors, _select_regressor, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_BASELINE, PRODUCTION_COST_SEED, enrich_supervised_for_production


def reproduce_production(data: pd.DataFrame, start: int, end: int, test_end: int, experiment_id: str):
    raw = data.copy(); raw["completion_year"] = pd.to_numeric(raw.completion_year, errors="coerce")
    raw_train, _ = temporal_project_split(raw, start, end, test_end)
    audit = audit_features(raw_train, CANDIDATE_FEATURES, minimum_availability=10, minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES), leakage_risks={
            "revised_cost_cr": "same official snapshot; production ablation contract retained",
            "cost_escalation_percentage": "same-snapshot revised cost derivative; production contract retained",
        })
    production_features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    cost_name, _base_cost, cost_selection = _select_regressor(raw_train, production_features, "actual_cost_overrun_percentage", 26203)
    delay_name, delay_model, delay_selection = _select_regressor(raw_train, production_features, "actual_delay_days", 26204)
    enriched = enrich_supervised_for_production(raw.copy()); enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train, _ = temporal_project_split(enriched, start, end, test_end)
    usable, trajectory_audit = usable_exp12_features(train)
    cost_added, cost_group, cost_feature_selection = select_exp12_features(train, production_features, usable, "actual_cost_overrun_percentage", cost_name, PRODUCTION_COST_SEED)
    cost_features = list(dict.fromkeys(production_features + cost_added))
    cost_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[cost_name], train, cost_features, "actual_cost_overrun_percentage")
    metadata = {"features_used": production_features, "cost_features_used": cost_features, "delay_features_used": production_features,
        "risk_features_used": production_features, "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        "production_cost_baseline": PRODUCTION_COST_BASELINE, "cost_trajectory_feature_group": cost_group, "cost_trajectory_features": cost_added}
    bundle = {"cost": cost_model, "delay": delay_model, "metadata": metadata}
    receipt = {"run_id": f"{experiment_id}-controlled-{start}-{end}", "selected_algorithms": {"cost": cost_name, "delay": delay_name}}
    reproduction = {"production_cost_baseline": PRODUCTION_COST_BASELINE, "production_features": production_features, "cost_features": cost_features,
        "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "cost_trajectory_feature_group": cost_group,
        "cost_trajectory_features": cost_added, "trajectory_feature_availability": trajectory_audit,
        "ordinary_algorithm_selection": {"cost": cost_selection, "delay": delay_selection}, "cost_trajectory_selection": cost_feature_selection}
    return bundle, receipt, reproduction


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment-id", required=True); parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True); parser.add_argument("--test-end", type=int, default=None); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, _identity = build_training_dataset(); data = data.copy(); data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    latest = int(data.completion_year.dropna().max()); test_end = int(args.test_end or latest)
    if test_end <= args.end: raise ValueError(f"test_end={test_end} must be after training_end={args.end}")
    adapter = get_experiment_adapter(args.experiment_id)
    production_bundle, production_receipt, reproduction = reproduce_production(data, args.start, args.end, test_end, args.experiment_id)
    result = adapter.module.fit_against_production(data=data, training_start=args.start, training_end=args.end, test_end=test_end,
        production_bundle=production_bundle, production_receipt=production_receipt)
    payload = _json_safe({"experiment": result["experiment"], "overall_comparison": result["overall_comparison"],
        "production_reproduction": reproduction, "window": {"training_start": args.start, "training_end": args.end, "testing_end": test_end}})
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("NEURAL_SEQUENCE_COMPARISON=" + json.dumps(payload["overall_comparison"], sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
