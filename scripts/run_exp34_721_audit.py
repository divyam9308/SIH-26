"""Re-evaluate Experiment 34 Delay on the exact 721-project production-comparable cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    _blend_predict,
    enrich_path_dependence,
    fit_experiment,
)
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)
from scripts.run_fast_current_experiment import fast_current_production


def gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def main() -> None:
    start, end, test_end = 2001, 2021, 2025
    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")

    production_bundle, production_receipt = fast_current_production(data, start, end, test_end)
    fitted = fit_experiment(
        data=data,
        training_start=start,
        training_end=end,
        test_end=test_end,
        production_bundle=production_bundle,
        production_receipt=production_receipt,
    )

    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    _train, test = temporal_project_split(enriched, start, end, test_end)

    cohort = _production_cost_evaluation_rows(test)
    project_count = int(cohort["canonical_project_id"].nunique())
    if project_count != 721:
        raise AssertionError(f"Expected exact 721-project comparable cohort, found {project_count}")

    contract = target_feature_contract(production_bundle["metadata"])
    production_prediction = np.maximum(
        0,
        production_bundle["delay"].predict(cohort[list(contract["delay"])]),
    )

    state = fitted["runtime_state"]
    experiment_prediction = np.maximum(
        0,
        _blend_predict(
            state["delay_models"],
            state["delay_weights"],
            cohort,
            state["delay_features"],
        ),
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

    improvement = gain(float(production_metrics["MAE"]), float(experiment_metrics["MAE"]))
    payload = {
        "training_window": "2001_2021",
        "testing_window": "2022_2025",
        "cohort_policy": "exact Exp12 production-comparable cohort",
        "projects": project_count,
        "snapshots": int(len(cohort)),
        "production_delay_mae": production_metrics["MAE"],
        "exp34_delay_mae": experiment_metrics["MAE"],
        "delay_improvement_percentage": round(improvement, 4),
        "delay_blend_weights": state["delay_weights"],
        "original_full_holdout_projects": fitted["overall_comparison"]["comparison_test_projects"],
        "original_full_holdout_exp34_delay_mae": fitted["overall_comparison"]["experiment_delay_mae"],
        "verdict": "BETTER" if improvement > 0 else "WORSE_OR_EQUAL",
    }

    output = Path("audit_outputs/exp34_delay_721.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("EXP34_721_AUDIT=" + json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
