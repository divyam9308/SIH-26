"""Experiment G: learned nonlinear earned-Cost benchmark.

Learn the empirical expenditure-vs-physical-progress curve from training projects
instead of imposing fixed EVM identities. The candidate and its internal control
use the same preregistered L1 LightGBM; the only control-vs-candidate difference
is the learned earned-Cost representation. Production remains untouched.
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

EXPERIMENT_ID = "exp_g"
EXPERIMENT_NAME = "G — learned nonlinear earned-Cost benchmark"
SEED = 13601
BASE_FEATURES = [
    "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr",
    "expenditure_ratio", "physical_progress", "financial_progress",
    "duration_ratio", "elapsed_duration_days", "planned_duration_days",
    "schedule_slippage_days", "progress_deviation", "cost_escalation_percentage",
    "exp12_cost_velocity_12m", "exp12_expenditure_velocity_6m",
    "exp12_expenditure_acceleration", "exp34_cost_revision_count",
]
EARNED_FEATURES = [
    "exp_g_expected_spend_ratio",
    "exp_g_spend_vs_norm",
    "exp_g_actual_cost_efficiency",
    "exp_g_norm_cost_efficiency",
    "exp_g_efficiency_gap",
    "exp_g_financial_effective_progress",
    "exp_g_progress_financial_mismatch",
]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _first_numeric(frame: pd.DataFrame, *cols: str) -> pd.Series:
    """Return the first available same-snapshot numeric alias, filling gaps causally."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for col in cols:
        if col not in frame.columns:
            continue
        values = _num(frame, col)
        out = out.where(out.notna(), values)
    return out


def _norm_sector(frame: pd.DataFrame) -> pd.Series:
    if "_norm_sector" in frame:
        return frame["_norm_sector"].astype("string").fillna("<NA>")
    return frame.get("sector", pd.Series("<NA>", index=frame.index)).astype("string").fillna("<NA>").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def _curve_inputs(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return canonical as-of physical progress and spend ratio.

    PAIMANA ingestion and production enrichment use several names for the same
    observable snapshot fields. Resolve only those same-snapshot aliases; never
    use outcomes, future rows, or fabricated defaults.
    """
    progress = _first_numeric(
        frame,
        "physical_progress",
        "physical_progress_percentage",
        "physical_progress_pct",
    )
    reconstructed_progress = _num(frame, "expected_progress_percentage") + _num(frame, "progress_deviation")
    progress = progress.where(progress.notna(), reconstructed_progress).clip(0, 100)

    spend = _first_numeric(frame, "expenditure_ratio", "financial_progress")
    cumulative = _first_numeric(
        frame,
        "cumulative_expenditure_cr",
        "current_expenditure_cr",
        "current_expenditure",
        "expenditure_cr",
    )
    approved = _first_numeric(frame, "approved_cost_cr", "original_cost", "original_cost_cr")
    reconstructed_spend = (cumulative / approved).where(approved.gt(0))
    spend = spend.where(spend.notna(), reconstructed_spend)
    # Financial progress is percentage-like in source data; expenditure_ratio is ratio-like.
    if spend.dropna().median() > 2.0:
        spend = spend / 100.0
    return progress, spend


def _fit_curve(train: pd.DataFrame) -> dict:
    work = train.copy()
    progress, spend = _curve_inputs(work)
    work["_progress_bin"] = (np.floor(progress / 5.0) * 5.0).astype("Int64")
    work["_sector"] = _norm_sector(work)
    work["_spend"] = spend
    global_curve = work.groupby("_progress_bin", dropna=True)["_spend"].agg(["median", "count"]).reset_index().sort_values("_progress_bin")
    sector_curve = work.groupby(["_sector", "_progress_bin"], dropna=False)["_spend"].agg(["median", "count"]).reset_index()
    sector_curve = sector_curve.loc[sector_curve["count"] >= 20].copy()
    if global_curve.empty:
        raise ValueError("Experiment G requires physical-progress/expenditure history")
    return {"global": global_curve, "sector": sector_curve}


def attach_earned_cost(train_reference: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    curves = _fit_curve(train_reference)
    out = score.copy()
    progress, spend = _curve_inputs(out)
    out["_progress_bin"] = (np.floor(progress / 5.0) * 5.0).astype("Int64")
    out["_sector"] = _norm_sector(out)
    global_curve = curves["global"].rename(columns={"median": "_global_spend"})
    sector_curve = curves["sector"].rename(columns={"median": "_sector_spend"})
    out = out.merge(global_curve[["_progress_bin", "_global_spend"]], on="_progress_bin", how="left", sort=False)
    out = out.merge(sector_curve[["_sector", "_progress_bin", "_sector_spend"]], on=["_sector", "_progress_bin"], how="left", sort=False)
    expected = pd.to_numeric(out["_sector_spend"], errors="coerce").fillna(pd.to_numeric(out["_global_spend"], errors="coerce"))
    progress_fraction = (progress / 100.0).clip(lower=0.02)
    out["exp_g_expected_spend_ratio"] = expected
    out["exp_g_spend_vs_norm"] = spend - expected
    out["exp_g_actual_cost_efficiency"] = spend / progress_fraction
    out["exp_g_norm_cost_efficiency"] = expected / progress_fraction
    out["exp_g_efficiency_gap"] = out["exp_g_actual_cost_efficiency"] - out["exp_g_norm_cost_efficiency"]

    gc = curves["global"]
    curve_spend = pd.to_numeric(gc["median"], errors="coerce").to_numpy(float)
    curve_progress = pd.to_numeric(gc["_progress_bin"], errors="coerce").to_numpy(float) / 100.0
    effective = np.full(len(out), np.nan, dtype=float)
    valid_curve = np.isfinite(curve_spend) & np.isfinite(curve_progress)
    for i, value in enumerate(spend.to_numpy(float)):
        if np.isfinite(value) and valid_curve.any():
            j = int(np.argmin(np.abs(curve_spend[valid_curve] - value)))
            effective[i] = curve_progress[valid_curve][j]
    out["exp_g_financial_effective_progress"] = effective
    out["exp_g_progress_financial_mismatch"] = progress_fraction.to_numpy(float) - effective
    return out.drop(columns=["_progress_bin", "_sector", "_global_spend", "_sector_spend"], errors="ignore")


def _design(train: pd.DataFrame, score: pd.DataFrame, features: list[str]):
    cols, left, right = [], {}, {}
    for col in features:
        if col not in train.columns or col not in score.columns:
            continue
        a = _num(train, col).replace([np.inf, -np.inf], np.nan)
        b = _num(score, col).replace([np.inf, -np.inf], np.nan)
        if not a.notna().any():
            continue
        median = float(a.median())
        cols.append(col); left[col] = a.fillna(median); right[col] = b.fillna(median)
    return cols, pd.DataFrame(left, index=train.index), pd.DataFrame(right, index=score.index)


def _model(seed: int):
    return LGBMRegressor(objective="regression_l1", n_estimators=450, learning_rate=0.025, max_depth=4, num_leaves=16, min_child_samples=60, subsample=0.9, colsample_bytree=0.9, reg_alpha=4, reg_lambda=20, random_state=seed, verbosity=-1, n_jobs=2)


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str], seed: int):
    cols, x_train, x_score = _design(train, score, features)
    target = _num(train, "actual_cost_overrun_percentage")
    valid = target.notna()
    model = _model(seed)
    model.fit(x_train.loc[valid], target.loc[valid].to_numpy(float), sample_weight=_num(train.loc[valid], "sample_weight").fillna(0).to_numpy(float))
    return np.asarray(model.predict(x_score), float), cols


def fit_experiment(training_end: int, output: str):
    ctx = prepare_context(training_end)
    train = ctx["train"].copy(); score = ctx["cohort"].copy()
    train_enriched = attach_earned_cost(train, train)
    score_enriched = attach_earned_cost(train, score)
    control, control_cols = _fit_predict(train_enriched, score_enriched, BASE_FEATURES, SEED)
    candidate, candidate_cols = _fit_predict(train_enriched, score_enriched, BASE_FEATURES + EARNED_FEATURES, SEED)

    pc = np.asarray(ctx["production_cost"], float); pdly = np.asarray(ctx["production_delay"], float)
    pcm = metric(score, "actual_cost_overrun_percentage", pc); ecm = metric(score, "actual_cost_overrun_percentage", candidate); control_mae = metric(score, "actual_cost_overrun_percentage", control); pdm = metric(score, "actual_delay_days", pdly); improvement = gain(pcm, ecm)
    evidence = score[["canonical_project_id", "sample_weight", "actual_cost_overrun_percentage"]].copy(); evidence["production"] = pc; evidence["experiment"] = candidate
    bootstrap = paired_project_mae_comparison(evidence, actual="actual_cost_overrun_percentage", baseline_prediction="production", candidate_prediction="experiment", bootstrap_samples=5000, seed=SEED)
    result = {
        "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": "cost",
        "training_start": 2001, "training_end": training_end, "test_start": ctx["test_start"], "test_end": ctx["test_end"],
        "production_cost_mae": pcm, "experiment_cost_mae": ecm, "cost_improvement_percentage": round(improvement, 6),
        "production_delay_mae": pdm, "experiment_delay_mae": pdm, "delay_improvement_percentage": 0.0,
        "comparison_test_projects": int(score["canonical_project_id"].nunique()), "comparison_test_snapshots": len(score),
        "delay_predictions_identical": True, "holdout_used_for_selection": False, "promotion_allowed": False,
        "paired_project_cost_bootstrap": bootstrap, "execution_verdict": "EXECUTION VALID",
        "scientific_verdict": "PROMOTION CANDIDATE" if improvement > 0 else "DO NOT PROMOTE",
        "details": {
            "changed_dimension": "learned_nonlinear_earned_cost_representation",
            "internal_same_model_control_mae": control_mae,
            "control_features": control_cols,
            "candidate_features": candidate_cols,
            "earned_features": EARNED_FEATURES,
            "curve_fit_uses_cost_outcome": False,
            "future_holdout_used_for_curve": False,
            "curve_input_fallbacks_use_same_snapshot_components_only": True,
        },
    }
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"EXP_G_PRODUCTION_COST_MAE={pcm:.6f}"); print(f"EXP_G_CONTROL_COST_MAE={control_mae:.6f}"); print(f"EXP_G_EXPERIMENT_COST_MAE={ecm:.6f}")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--end", type=int, choices=[2021,2022], required=True); p.add_argument("--output", required=True); a=p.parse_args(); fit_experiment(a.end,a.output)


if __name__ == "__main__":
    main()
