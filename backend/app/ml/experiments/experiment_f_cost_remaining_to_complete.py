"""Experiment F: remaining Cost-to-complete forecasting.

Instead of predicting final overrun percentage directly, fit the money still
required to finish the project, normalized by approved cost.  The final Cost is
reconstructed as current cumulative expenditure plus predicted remaining spend,
which enforces final expenditure >= expenditure already incurred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.post_exp113_delay_common import gain, metric, prepare_context
from backend.app.ml.monthly_training import _json_safe

EXPERIMENT_ID = "exp_f"
EXPERIMENT_NAME = "F — remaining Cost-to-complete"
SEED = 13501
FEATURES = [
    "approved_cost_cr",
    "revised_cost_cr",
    "cumulative_expenditure_cr",
    "expenditure_ratio",
    "physical_progress",
    "financial_progress",
    "duration_ratio",
    "elapsed_duration_days",
    "planned_duration_days",
    "schedule_slippage_days",
    "progress_deviation",
    "cost_escalation_percentage",
    "exp12_cost_velocity_12m",
    "exp12_expenditure_velocity_6m",
    "exp12_expenditure_acceleration",
    "exp34_cost_revision_count",
    "exp34_cumulative_abs_cost_revision_pct",
]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def remaining_cost_ratio_target(frame: pd.DataFrame) -> pd.Series:
    approved = _num(frame, "approved_cost_cr")
    spent = _num(frame, "cumulative_expenditure_cr")
    final = _num(frame, "reported_completion_expenditure_cr")
    remaining = (final - spent).clip(lower=0.0)
    return remaining / approved.where(approved > 0)


def _design(train: pd.DataFrame, score: pd.DataFrame):
    cols, medians, left, right = [], {}, {}, {}
    for col in FEATURES:
        if col not in train.columns or col not in score.columns:
            continue
        a = _num(train, col).replace([np.inf, -np.inf], np.nan)
        b = _num(score, col).replace([np.inf, -np.inf], np.nan)
        if not a.notna().any():
            continue
        m = float(a.median())
        cols.append(col)
        medians[col] = m
        left[col] = a.fillna(m)
        right[col] = b.fillna(m)
    if not cols:
        raise ValueError("Experiment F found no usable as-of numeric features")
    return cols, medians, pd.DataFrame(left, index=train.index), pd.DataFrame(right, index=score.index)


def reconstruct_cost_overrun(score: pd.DataFrame, predicted_remaining_ratio: np.ndarray) -> np.ndarray:
    approved = _num(score, "approved_cost_cr").to_numpy(float)
    spent = _num(score, "cumulative_expenditure_cr").fillna(0.0).to_numpy(float)
    remaining = np.maximum(0.0, np.asarray(predicted_remaining_ratio, dtype=float)) * approved
    final_cost = np.maximum(spent, spent + remaining)
    with np.errstate(divide="ignore", invalid="ignore"):
        overrun = (final_cost / approved - 1.0) * 100.0
    return np.where(np.isfinite(overrun), overrun, 0.0)


def _persist(ctx: dict, prediction: np.ndarray, details: dict, output: str):
    c = ctx["cohort"]
    pc = np.asarray(ctx["production_cost"], dtype=float)
    pdly = np.asarray(ctx["production_delay"], dtype=float)
    ec = np.asarray(prediction, dtype=float)
    pcm = metric(c, "actual_cost_overrun_percentage", pc)
    ecm = metric(c, "actual_cost_overrun_percentage", ec)
    pdm = metric(c, "actual_delay_days", pdly)
    improvement = gain(pcm, ecm)
    evidence = c[["canonical_project_id", "sample_weight", "actual_cost_overrun_percentage"]].copy()
    evidence["production"] = pc
    evidence["experiment"] = ec
    bootstrap = paired_project_mae_comparison(
        evidence,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production",
        candidate_prediction="experiment",
        bootstrap_samples=5000,
        seed=SEED,
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "scope": "cost",
        "training_start": 2001,
        "training_end": ctx["training_end"],
        "test_start": ctx["test_start"],
        "test_end": ctx["test_end"],
        "production_cost_mae": pcm,
        "experiment_cost_mae": ecm,
        "cost_improvement_percentage": round(improvement, 6),
        "production_delay_mae": pdm,
        "experiment_delay_mae": pdm,
        "delay_improvement_percentage": 0.0,
        "comparison_test_projects": int(c["canonical_project_id"].nunique()),
        "comparison_test_snapshots": len(c),
        "delay_predictions_identical": True,
        "holdout_used_for_selection": False,
        "promotion_allowed": False,
        "paired_project_cost_bootstrap": bootstrap,
        "execution_verdict": "EXECUTION VALID",
        "scientific_verdict": "PROMOTION CANDIDATE" if improvement > 0 else "DO NOT PROMOTE",
        "details": _json_safe(details),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"EXP_F_PRODUCTION_COST_MAE={pcm:.6f}")
    print(f"EXP_F_EXPERIMENT_COST_MAE={ecm:.6f}")
    print(f"EXP_F_COST_IMPROVEMENT_PERCENT={improvement:.6f}")
    return result


def fit_experiment(training_end: int, output: str):
    ctx = prepare_context(training_end)
    train = ctx["train"].copy()
    score = ctx["cohort"].copy()
    target = remaining_cost_ratio_target(train)
    valid = target.notna() & np.isfinite(target) & (_num(train, "approved_cost_cr") > 0)
    train = train.loc[valid].copy()
    target = target.loc[valid].clip(lower=0.0)
    cols, medians, x_train, x_score = _design(train, score)
    cap = max(float(target.quantile(0.995)), 0.01)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=450,
        learning_rate=0.025,
        max_depth=4,
        num_leaves=16,
        min_child_samples=60,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=4.0,
        reg_lambda=20.0,
        random_state=SEED,
        verbosity=-1,
        n_jobs=2,
    )
    model.fit(x_train, np.log1p(target.to_numpy(float)), sample_weight=_num(train, "sample_weight").fillna(0.0).to_numpy(float))
    predicted_remaining = np.expm1(np.asarray(model.predict(x_score), dtype=float))
    predicted_remaining = np.clip(predicted_remaining, 0.0, cap)
    prediction = reconstruct_cost_overrun(score, predicted_remaining)
    details = {
        "changed_dimension": "remaining_cost_to_complete_target",
        "target": "log1p(max(final_completion_expenditure-current_expenditure,0)/approved_cost)",
        "reconstruction": "current expenditure + approved cost * predicted remaining ratio",
        "final_cost_below_current_spend_allowed": False,
        "target_cap_training_q995": cap,
        "features": cols,
        "feature_medians_fit_on_training_only": True,
        "training_rows": len(train),
    }
    return _persist(ctx, prediction, details, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    fit_experiment(a.end, a.output)


if __name__ == "__main__":
    main()
