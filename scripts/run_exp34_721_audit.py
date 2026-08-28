"""Re-evaluate Experiment 34 Delay on the exact 721-project production-comparable cohort.

This audit deliberately fits no Cost model. It reconstructs the exact current
production Delay feature audit/model-family selection, fits Exp34 Delay, and
scores both on the identical 721-project future cohort.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES,
    _blend_predict,
    _fit_delay_family_models,
    _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
)
from backend.app.ml.monthly_training import (
    _regression_metrics,
    _select_regressor,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
)


def gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def main() -> None:
    start, end, test_end = 2001, 2021, 2025
    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")

    # Rebuild current production Delay exactly: same feature audit, seed,
    # temporal selection and final refit. No historical metric is reused.
    production_train, _production_test = temporal_project_split(data, start, end, test_end)
    audit = audit_features(
        production_train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage same-snapshot signal; evaluated by production ablation",
            "cost_escalation_percentage": "same-snapshot revised-cost derivative; evaluated by production ablation",
        },
    )
    production_features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    production_delay_family, production_delay_model, production_delay_comparisons = _select_regressor(
        production_train,
        production_features,
        "actual_delay_days",
        26204,
    )

    # Exp34 uses the same training boundary plus causal path features.
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    experiment_train, test = temporal_project_split(enriched, start, end, test_end)
    delay_features = list(dict.fromkeys(production_features + PATH_FEATURES))
    delay_weights, delay_oof = _oof_delay_weights(experiment_train, delay_features)
    delay_models = _fit_delay_family_models(experiment_train, delay_features)

    # Exact 721-project production-comparable cohort. This assertion is the key
    # contract requested for the re-evaluation.
    cohort = _production_cost_evaluation_rows(test)
    project_count = int(cohort["canonical_project_id"].nunique())
    if project_count != 721:
        raise AssertionError(f"Expected exact 721-project comparable cohort, found {project_count}")

    production_prediction = np.maximum(
        0,
        production_delay_model.predict(cohort[production_features]),
    )
    experiment_prediction = np.maximum(
        0,
        _blend_predict(delay_models, delay_weights, cohort, delay_features),
    )

    production_metrics = _regression_metrics(
        cohort["actual_delay_days"],
        production_prediction,
        cohort["sample_weight"],
        cohort["canonical_project_id"],
    )
    experiment_metrics = _regression_metrics(
        cohort["actual_delay_days"],
        experiment_prediction,
        cohort["sample_weight"],
        cohort["canonical_project_id"],
    )

    # Also reproduce the original 728-project Exp34 result as a diagnostic only;
    # it does not drive the 721-project verdict.
    full_experiment_prediction = np.maximum(
        0,
        _blend_predict(delay_models, delay_weights, test, delay_features),
    )
    full_experiment_metrics = _regression_metrics(
        test["actual_delay_days"],
        full_experiment_prediction,
        test["sample_weight"],
        test["canonical_project_id"],
    )

    improvement = gain(float(production_metrics["MAE"]), float(experiment_metrics["MAE"]))
    payload = {
        "training_window": "2001_2021",
        "testing_window": "2022_2025",
        "cohort_policy": "exact Exp12 production-comparable cohort",
        "projects": project_count,
        "snapshots": int(len(cohort)),
        "production_delay_family": production_delay_family,
        "production_delay_family_comparisons": production_delay_comparisons,
        "production_delay_mae": production_metrics["MAE"],
        "exp34_delay_mae": experiment_metrics["MAE"],
        "delay_improvement_percentage": round(improvement, 4),
        "delay_blend_weights": delay_weights,
        "rolling_oof": delay_oof,
        "diagnostic_full_holdout_projects": int(test["canonical_project_id"].nunique()),
        "diagnostic_full_holdout_exp34_delay_mae": full_experiment_metrics["MAE"],
        "verdict": "BETTER" if improvement > 0 else "WORSE_OR_EQUAL",
    }

    output = Path("audit_outputs/exp34_delay_721.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("EXP34_721_AUDIT=" + json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
