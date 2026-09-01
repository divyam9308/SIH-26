"""Exp92: causal local-level/local-trend state-space Cost signals."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backend.app.ml.experiments.post_u1_cost_common import (
    current_cost_oof,
    fit_residual_booster,
    persist,
    prepare_context,
    window_contract,
)

EXPERIMENT_ID = "exp_92"
EXPERIMENT_NAME = "Dynamic state-space Cost forecast"
EXPERIMENT_SCOPE = "cost"
EXPERIMENT_SEQUENCE = 92
ALPHA = 0.35
BETA = 0.08
STATE_FEATURES = [
    "exp92_level",
    "exp92_trend",
    "exp92_innovation",
    "exp92_innovation_sd",
    "exp92_terminal_projection",
]


def _state(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal state features while preserving the input row order.

    Cohort frames retain source-data index labels, which are not guaranteed to be
    contiguous or zero-based. State arrays therefore must never be indexed with
    those labels. Resetting to positional row ids before the chronological sort
    makes the filter safe for both OOF frames and real evaluation cohorts.
    """

    x = frame.copy().reset_index(drop=True)
    x["_exp92_row_order"] = np.arange(len(x), dtype=int)
    x["_exp92_date"] = pd.to_datetime(x["snapshot_date"], errors="coerce")
    x = x.sort_values(
        ["canonical_project_id", "_exp92_date", "_exp92_row_order"],
        kind="stable",
    ).reset_index(drop=True)

    for feature in STATE_FEATURES:
        x[feature] = np.nan

    for _, group in x.groupby("canonical_project_id", sort=False, dropna=False):
        level = None
        trend = 0.0
        innovation_variance = 0.0

        for idx in group.index:
            value = pd.to_numeric(x.at[idx, "cost_escalation_percentage"], errors="coerce")
            if pd.isna(value):
                continue
            value = float(value)

            if level is None:
                level = value
                trend = 0.0
                innovation = 0.0
                innovation_variance = 0.0
            else:
                predicted = level + trend
                innovation = value - predicted
                level = predicted + ALPHA * innovation
                trend = trend + BETA * innovation
                innovation_variance = 0.8 * innovation_variance + 0.2 * innovation**2

            x.at[idx, "exp92_level"] = level
            x.at[idx, "exp92_trend"] = trend
            x.at[idx, "exp92_innovation"] = innovation
            x.at[idx, "exp92_innovation_sd"] = np.sqrt(max(innovation_variance, 0.0))
            x.at[idx, "exp92_terminal_projection"] = level + 6.0 * trend

    return (
        x.sort_values("_exp92_row_order", kind="stable")
        .drop(columns=["_exp92_row_order", "_exp92_date"])
        .reset_index(drop=True)
    )


def fit_experiment(training_end, output):
    window_contract(training_end)
    ctx = prepare_context(training_end)

    oof = _state(current_cost_oof(ctx["train"], ctx["cost_model"]))
    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_cost"]
    score = _state(score)

    features = [
        "production_prediction",
        "exp92_level",
        "exp92_trend",
        "exp92_innovation",
        "exp92_innovation_sd",
        "exp92_terminal_projection",
        "duration_ratio",
        "expenditure_ratio",
        "progress_deviation",
    ]
    corr, meta = fit_residual_booster(oof, score, features, 9201)
    meta.update(
        {
            "state_alpha": ALPHA,
            "state_beta": BETA,
            "state_filter": "causal local linear trend",
        }
    )
    return persist(
        EXPERIMENT_ID,
        EXPERIMENT_NAME,
        ctx,
        ctx["production_cost"] + corr,
        meta,
        output,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2019, 2021], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_experiment(args.end, args.output)


if __name__ == "__main__":
    main()
