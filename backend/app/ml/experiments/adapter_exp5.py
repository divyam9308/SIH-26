"""Retrain & Compare adapter for Experiment 5 common-holdout lifecycle model.

Experiment 5 preserves Krish's audited 25-feature lifecycle implementation. The
comparison contract is updated for the current repository: production cost is
the promoted Experiment 12 trajectory baseline, while both sides are scored on
the exact same 2022-2025 cohort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml import monthly_training as trainer
from backend.app.ml.experiments.common_holdout_training_window import audited_feature_contract
from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract
from backend.app.ml.provenance import frame_fingerprint, new_run_id

EXPERIMENT_ID = "exp_5"
EXPERIMENT_SEQUENCE = 5
EXPERIMENT_NAME = "Common-holdout audited lifecycle model"
EXPERIMENT_SCOPE = "cost_delay"
TEST_START = 2022
TEST_END = 2025
SEEDS = {2019: 26519, 2021: 26521}


def _metric(frame: pd.DataFrame, actual: str, prediction: str) -> dict:
    return trainer._regression_metrics(
        frame[actual],
        frame[prediction].to_numpy(),
        frame.sample_weight,
        frame.canonical_project_id,
    )


def _improvement(baseline: float, candidate: float):
    return round((baseline - candidate) / baseline * 100.0, 4) if baseline else None


def _stage_metrics(compare: pd.DataFrame, prefix: str) -> dict:
    result = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = compare[compare.lifecycle_stage.eq(stage)]
        if part.empty:
            result[stage] = {"available": False}
            continue
        result[stage] = {
            "available": True,
            "cost": _metric(part, "actual_cost_overrun_percentage", f"{prefix}_cost"),
            "delay": _metric(part, "actual_delay_days", f"{prefix}_delay"),
        }
    return result


def fit_against_production(
    *,
    data,
    training_start,
    training_end,
    test_end,
    production_bundle,
    production_receipt,
    **_kwargs,
):
    start = int(training_start)
    end = int(training_end)
    if end not in SEEDS:
        raise ValueError("Experiment 5 comparison is defined for training ends 2019 and 2021 only.")

    frozen = enrich_supervised_for_production(data.copy())
    frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce")
    frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    train = frozen[frozen.completion_year.between(start, end)].copy()
    compare = frozen[frozen.completion_year.between(TEST_START, TEST_END)].copy()
    if train.empty or compare.empty:
        raise ValueError("Experiment 5 requires non-empty training and 2022-2025 holdout cohorts.")
    overlap = set(train.canonical_project_id.dropna()) & set(compare.canonical_project_id.dropna())
    if overlap:
        raise ValueError(f"Experiment 5 train/common-holdout project overlap: {len(overlap)}")

    features, audit = audited_feature_contract(train)
    seed = SEEDS[end]
    candidate_bundle, _candidate_metrics, _candidate_rows = trainer._train_variant(
        train, compare, features, seed
    )

    production_contract = target_feature_contract(production_bundle["metadata"])
    compare["production_cost"] = production_bundle["cost"].predict(compare[production_contract["cost"]])
    compare["production_delay"] = np.maximum(
        0, production_bundle["delay"].predict(compare[production_contract["delay"]])
    )
    compare["experiment_cost"] = candidate_bundle["models"]["cost"].predict(compare[features])
    compare["experiment_delay"] = np.maximum(
        0, candidate_bundle["models"]["delay"].predict(compare[features])
    )

    production_cost = _metric(compare, "actual_cost_overrun_percentage", "production_cost")
    experiment_cost = _metric(compare, "actual_cost_overrun_percentage", "experiment_cost")
    production_delay = _metric(compare, "actual_delay_days", "production_delay")
    experiment_delay = _metric(compare, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(
        compare,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost",
        candidate_prediction="experiment_cost",
    )
    paired_delay = paired_project_mae_comparison(
        compare,
        actual="actual_delay_days",
        baseline_prediction="production_delay",
        candidate_prediction="experiment_delay",
    )

    prod_stage = _stage_metrics(compare, "production")
    exp_stage = _stage_metrics(compare, "experiment")
    run_id = new_run_id()
    overall = {
        "production_cost_mae": production_cost["MAE"],
        "experiment_cost_mae": experiment_cost["MAE"],
        "absolute_mae_improvement_pp": round(production_cost["MAE"] - experiment_cost["MAE"], 4),
        "improvement_percentage": _improvement(production_cost["MAE"], experiment_cost["MAE"]),
        "production_delay_mae": production_delay["MAE"],
        "experiment_delay_mae": experiment_delay["MAE"],
        "delay_absolute_mae_improvement_days": round(production_delay["MAE"] - experiment_delay["MAE"], 4),
        "delay_improvement_percentage": _improvement(production_delay["MAE"], experiment_delay["MAE"]),
        "production_cost_metrics": production_cost,
        "experiment_cost_metrics": experiment_cost,
        "production_delay_metrics": production_delay,
        "experiment_delay_metrics": experiment_delay,
        "paired_project_comparison": paired_cost,
        "paired_project_cost_comparison": paired_cost,
        "paired_project_delay_comparison": paired_delay,
        "production_stage_metrics": prod_stage,
        "experiment_stage_metrics": exp_stage,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(compare)),
        "comparison_test_period": [TEST_START, TEST_END],
        "common_holdout_fingerprint": frame_fingerprint(
            compare[["canonical_project_id", "snapshot_date", "completion_year", "sample_weight"]]
        ),
        "production_cost_baseline": (production_bundle.get("metadata") or {}).get("production_cost_baseline"),
        "production_cost_feature_count": len(production_contract["cost"]),
        "production_delay_feature_count": len(production_contract["delay"]),
        "experiment_feature_count": len(features),
        "experiment_seed": seed,
        "seed_interpretation": (
            "Exp5 preserves Krish's original seeds; delay differences are therefore not a clean architectural effect because Exp5 does not define a distinct delay architecture."
        ),
    }

    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "model_role": "experiment",
        "scope": EXPERIMENT_SCOPE,
        "decision": "PENDING",
        "promotion_allowed": False,
        "training_period": [start, end],
        "testing_period": [TEST_START, TEST_END],
        "feature_count": len(features),
        "features_used": features,
        "selected_algorithms": candidate_bundle["selected_algorithms"],
        "seed": seed,
        "feature_audit": audit,
        "metrics": {"cost": experiment_cost, "delay": experiment_delay},
    }
    return {
        "experiment": experiment,
        "overall_comparison": overall,
        "runtime_state": {
            "models": candidate_bundle["models"],
            "features": features,
            "common_projects": set(compare.canonical_project_id.astype(str)),
            "test_period": (TEST_START, TEST_END),
            "selected_algorithms": candidate_bundle["selected_algorithms"],
            "seed": seed,
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    years = pd.to_numeric(frame.completion_year, errors="coerce")
    projects = frame.canonical_project_id.astype(str)
    start, end = state["test_period"]
    return frame[years.between(start, end) & projects.isin(state["common_projects"])].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    candidate = row.to_frame().T.reindex(columns=state["features"])
    return {
        "predicted_cost_overrun": round(float(state["models"]["cost"].predict(candidate)[0]), 4),
        "predicted_delay_days": round(float(max(0.0, state["models"]["delay"].predict(candidate)[0])), 4),
        "predicted_risk": str(state["models"]["risk"].predict(candidate)[0]),
        "selected_algorithms": state["selected_algorithms"],
        "experiment_seed": state["seed"],
        "comparison_test_period": list(state["test_period"]),
    }
