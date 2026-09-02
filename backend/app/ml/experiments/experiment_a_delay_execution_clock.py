"""Experiment A: learned nonlinear execution-clock features for Delay.

The experiment learns a training-only empirical relationship between calendar
lifecycle position and observed physical/financial execution.  It then measures
how far each project is ahead/behind that learned clock and feeds those signals
into a bounded residual correction above the current Exp113 production Delay
forecast.  The final 2022-2025 holdout is evaluation-only.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from backend.app.ml.experiments.post_exp113_delay_common import (
    fit_residual,
    persist,
    prepare_context,
    production_oof,
)

EXPERIMENT_ID = "exp_a"
EXPERIMENT_NAME = "A — learned nonlinear execution clock"
EXPERIMENT_SCOPE = "delay"
SEED = 13001
CLOCK_FEATURES = [
    "exp_a_expected_progress",
    "exp_a_expected_spend_ratio",
    "exp_a_progress_clock_gap",
    "exp_a_spend_clock_gap",
    "exp_a_physical_effective_ratio",
    "exp_a_financial_effective_ratio",
    "exp_a_physical_clock_lag",
    "exp_a_financial_clock_lag",
    "exp_a_clock_disagreement",
]
RESIDUAL_FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "physical_progress",
    "expenditure_ratio",
    "progress_deviation",
    "exp58_delay_hier_prior",
    "exp58_group_support",
    *CLOCK_FEATURES,
]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _clock_bin(values: pd.Series) -> pd.Series:
    clipped = _num(pd.DataFrame({"x": values}), "x").clip(lower=0.0, upper=3.0)
    return np.floor(clipped * 10.0).astype("Int64")


def _fit_clock(reference: pd.DataFrame) -> dict:
    ref = reference.copy()
    ref["_clock_bin"] = _clock_bin(_num(ref, "duration_ratio"))
    ref["_norm_sector"] = ref.get("_norm_sector", ref.get("sector", "<NA>")).astype("string").fillna("<NA>")
    ref["_progress"] = _num(ref, "physical_progress")
    ref["_spend"] = _num(ref, "expenditure_ratio")

    global_curve = (
        ref.groupby("_clock_bin", dropna=True)
        .agg(progress=("_progress", "median"), spend=("_spend", "median"), support=("_progress", "count"))
        .reset_index()
        .dropna(subset=["_clock_bin"])
        .sort_values("_clock_bin")
    )
    if global_curve.empty:
        raise ValueError("Experiment A requires observable duration/progress history")

    sector_curve = (
        ref.groupby(["_norm_sector", "_clock_bin"], dropna=False)
        .agg(progress=("_progress", "median"), spend=("_spend", "median"), support=("_progress", "count"))
        .reset_index()
    )
    sector_curve = sector_curve.loc[sector_curve["support"] >= 20].copy()
    return {"global": global_curve, "sector": sector_curve}


def _nearest_effective_ratio(observed: np.ndarray, curve_values: np.ndarray, curve_bins: np.ndarray) -> np.ndarray:
    out = np.full(len(observed), np.nan, dtype=float)
    good_curve = np.isfinite(curve_values) & np.isfinite(curve_bins)
    cv = curve_values[good_curve]
    cb = curve_bins[good_curve]
    if not len(cv):
        return out
    for i, value in enumerate(observed):
        if np.isfinite(value):
            out[i] = float(cb[int(np.argmin(np.abs(cv - value)))]) / 10.0
    return out


def attach_clock(reference: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    """Attach clock features using only ``reference`` rows to fit the clock."""
    curves = _fit_clock(reference)
    out = score.copy()
    out["_clock_bin"] = _clock_bin(_num(out, "duration_ratio"))
    out["_norm_sector"] = out.get("_norm_sector", out.get("sector", "<NA>")).astype("string").fillna("<NA>")

    global_curve = curves["global"].rename(columns={"progress": "_g_progress", "spend": "_g_spend"})
    sector_curve = curves["sector"].rename(columns={"progress": "_s_progress", "spend": "_s_spend"})
    out = out.merge(global_curve[["_clock_bin", "_g_progress", "_g_spend"]], on="_clock_bin", how="left", sort=False)
    out = out.merge(sector_curve[["_norm_sector", "_clock_bin", "_s_progress", "_s_spend"]], on=["_norm_sector", "_clock_bin"], how="left", sort=False)

    out["exp_a_expected_progress"] = _num(out, "_s_progress").fillna(_num(out, "_g_progress"))
    out["exp_a_expected_spend_ratio"] = _num(out, "_s_spend").fillna(_num(out, "_g_spend"))
    out["exp_a_progress_clock_gap"] = _num(out, "physical_progress") - out["exp_a_expected_progress"]
    out["exp_a_spend_clock_gap"] = _num(out, "expenditure_ratio") - out["exp_a_expected_spend_ratio"]

    gc = curves["global"]
    bins = pd.to_numeric(gc["_clock_bin"], errors="coerce").to_numpy(float)
    out["exp_a_physical_effective_ratio"] = _nearest_effective_ratio(
        _num(out, "physical_progress").to_numpy(float),
        pd.to_numeric(gc["progress"], errors="coerce").to_numpy(float),
        bins,
    )
    out["exp_a_financial_effective_ratio"] = _nearest_effective_ratio(
        _num(out, "expenditure_ratio").to_numpy(float),
        pd.to_numeric(gc["spend"], errors="coerce").to_numpy(float),
        bins,
    )
    out["exp_a_physical_clock_lag"] = _num(out, "duration_ratio") - out["exp_a_physical_effective_ratio"]
    out["exp_a_financial_clock_lag"] = _num(out, "duration_ratio") - out["exp_a_financial_effective_ratio"]
    out["exp_a_clock_disagreement"] = out["exp_a_physical_effective_ratio"] - out["exp_a_financial_effective_ratio"]
    return out.drop(columns=["_clock_bin", "_g_progress", "_g_spend", "_s_progress", "_s_spend"], errors="ignore")


def _forward_clock_oof(oof: pd.DataFrame) -> pd.DataFrame:
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in year_col.dropna().unique())
    parts = []
    for year in years[1:]:
        reference = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(reference) < 100 or validation.empty:
            continue
        parts.append(attach_clock(reference, validation))
    if not parts:
        raise ValueError("Experiment A has no forward clock folds")
    return pd.concat(parts, ignore_index=True)


def fit_experiment(training_end: int, output: str):
    ctx = prepare_context(training_end)
    oof = production_oof(ctx, max_folds=6)
    meta = _forward_clock_oof(oof)
    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_delay"]
    score = attach_clock(oof, score)
    correction, details = fit_residual(meta, score, RESIDUAL_FEATURES, SEED)
    details.update(
        {
            "changed_dimension": "learned_execution_clock",
            "clock_fit_for_meta": "strictly earlier production-OOF years",
            "holdout_clock_fit": "all allowed production-OOF training evidence",
            "target_columns_used_for_clock": False,
            "clock_features": CLOCK_FEATURES,
        }
    )
    return persist(EXPERIMENT_ID, EXPERIMENT_NAME, ctx, ctx["production_delay"] + correction, details, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_experiment(args.end, args.output)


if __name__ == "__main__":
    main()
