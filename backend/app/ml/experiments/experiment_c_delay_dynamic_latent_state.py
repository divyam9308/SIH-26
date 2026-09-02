"""Experiment C: causal multivariate dynamic execution state for Delay.

A low-capacity state-space representation is built from observable execution
signals only.  Every row is updated causally from the same project's current and
earlier reports; later reports cannot alter an earlier state.  The latent state
is used only as a residual signal above current Exp113 production.
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

EXPERIMENT_ID = "exp_c"
EXPERIMENT_NAME = "C — multivariate dynamic latent execution state"
SEED = 13201
ALPHA = 0.35
INPUTS = [
    "physical_progress",
    "expenditure_ratio",
    "schedule_slippage_days",
    "duration_ratio",
    "cost_escalation_percentage",
    "progress_deviation",
]
STATE_FEATURES = [
    "exp_c_latent_level",
    "exp_c_latent_trend",
    "exp_c_innovation",
    "exp_c_state_uncertainty",
    "exp_c_deterioration_state",
    "exp_c_deterioration_trend",
]
RESIDUAL_FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "expenditure_ratio",
    "progress_deviation",
    "exp58_delay_hier_prior",
    "exp58_group_support",
    *STATE_FEATURES,
]


def _fit_scaler(reference: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    medians, scales = {}, {}
    for col in INPUTS:
        values = pd.to_numeric(reference.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median()) if values.notna().any() else 0.0
        mad = float((values - median).abs().median()) if values.notna().any() else 1.0
        medians[col] = median
        scales[col] = max(1.4826 * mad, 1e-6)
    return medians, scales


def attach_dynamic_state(reference: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    """Fit robust scaling on reference and causally filter score trajectories."""
    medians, scales = _fit_scaler(reference)
    out = score.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    original_index = np.arange(len(out))
    out["_exp_c_order"] = original_index
    out = out.sort_values(["canonical_project_id", "snapshot_date", "_exp_c_order"], kind="mergesort").copy()

    zcols = []
    for col in INPUTS:
        z = (pd.to_numeric(out.get(col), errors="coerce") - medians[col]) / scales[col]
        name = f"_z_{col}"
        out[name] = z.fillna(0.0).clip(-8.0, 8.0)
        zcols.append(name)

    level_all = np.zeros(len(out), dtype=float)
    trend_all = np.zeros(len(out), dtype=float)
    innovation_all = np.zeros(len(out), dtype=float)
    uncertainty_all = np.zeros(len(out), dtype=float)
    deterioration_all = np.zeros(len(out), dtype=float)
    deterioration_trend_all = np.zeros(len(out), dtype=float)

    for _, positions in out.groupby("canonical_project_id", sort=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        x = out.iloc[pos][zcols].to_numpy(float)
        state = np.zeros(x.shape[1], dtype=float)
        variance = np.zeros(x.shape[1], dtype=float)
        prev_deterioration = 0.0
        for local, row in enumerate(x):
            previous = state.copy()
            innovation = row - previous
            state = ALPHA * row + (1.0 - ALPHA) * previous
            variance = ALPHA * innovation**2 + (1.0 - ALPHA) * variance
            trend = state - previous
            global_pos = pos[local]
            level_all[global_pos] = float(np.mean(state))
            trend_all[global_pos] = float(np.mean(trend))
            innovation_all[global_pos] = float(np.mean(np.abs(innovation)))
            uncertainty_all[global_pos] = float(np.mean(np.sqrt(np.maximum(variance, 0.0))))
            # Higher slippage/duration and lower progress/spend are adverse.
            deterioration = float(np.mean([state[2], state[3], -state[0], -state[1]]))
            deterioration_all[global_pos] = deterioration
            deterioration_trend_all[global_pos] = deterioration - prev_deterioration
            prev_deterioration = deterioration

    out["exp_c_latent_level"] = level_all
    out["exp_c_latent_trend"] = trend_all
    out["exp_c_innovation"] = innovation_all
    out["exp_c_state_uncertainty"] = uncertainty_all
    out["exp_c_deterioration_state"] = deterioration_all
    out["exp_c_deterioration_trend"] = deterioration_trend_all
    out = out.sort_values("_exp_c_order", kind="mergesort")
    return out.drop(columns=["_exp_c_order", *zcols], errors="ignore")


def _forward_state_oof(oof: pd.DataFrame) -> pd.DataFrame:
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in year_col.dropna().unique())
    parts = []
    for year in years[1:]:
        reference = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(reference) < 100 or validation.empty:
            continue
        parts.append(attach_dynamic_state(reference, validation))
    if not parts:
        raise ValueError("Experiment C has no forward latent-state folds")
    return pd.concat(parts, ignore_index=True)


def fit_experiment(training_end: int, output: str):
    ctx = prepare_context(training_end)
    oof = production_oof(ctx, max_folds=6)
    meta = _forward_state_oof(oof)
    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_delay"]
    score = attach_dynamic_state(oof, score)
    correction, details = fit_residual(meta, score, RESIDUAL_FEATURES, SEED)
    details.update(
        {
            "changed_dimension": "causal_multivariate_dynamic_state",
            "state_inputs": INPUTS,
            "filter_alpha": ALPHA,
            "state_scaler_for_meta": "strictly earlier OOF evidence",
            "future_row_invariance_required": True,
            "state_features": STATE_FEATURES,
        }
    )
    return persist(EXPERIMENT_ID, EXPERIMENT_NAME, ctx, ctx["production_delay"] + correction, details, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    fit_experiment(a.end, a.output)


if __name__ == "__main__":
    main()
