"""Exp126: Aditya reporting-cadence/data-quality delay challenger port.

Ports the successful delay-side idea from adityaab2007/SIH-26-Aditya PR #19
onto this repository's newer post-Exp113 leakage-safe comparison harness.
Only causal current/past snapshot behavior is engineered; production code and
cost predictions remain untouched.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import backend.app.ml.experiments.post_exp113_delay_common as _common
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as _fast_train_current_production,
)
from backend.app.ml.experiments.post_exp113_delay_common import (
    fit_residual,
    persist,
    prepare_context,
    production_oof,
)

# Branch-local execution substitution only: the fast wrapper calls the exact
# canonical Exp105+Exp113 trainer while parallelizing independent internal OOF
# work. This keeps the experiment/model/evaluation contract unchanged.
_common.train_current_production = _fast_train_current_production

EXPERIMENT_ID = "exp126"
NAME = "Reporting Cadence and Data-Quality Behavior (Aditya PR #19 port)"
FEATURES = [
    "is_report_gap_days",
    "is_report_gap_mean3",
    "is_report_gap_std6",
    "is_snapshot_index",
    "is_missing_count",
    "is_missing_delta",
    "is_unchanged_core_count",
    "is_stale_report_flag",
    "is_gap_x_duration",
    "is_missing_x_duration",
]
QUALITY_COLUMNS = [
    "revised_cost_cr",
    "cumulative_expenditure_cr",
    "schedule_slippage_days",
    "expenditure_ratio",
    "expected_progress_percentage",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def add_reporting_behavior_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Exp126 features without using any future snapshot information.

    Features are calculated independently within each project after sorting by
    snapshot date, then restored to the caller's original row order so residual
    corrections stay aligned with the production predictions.
    """
    if "canonical_project_id" not in frame.columns or "snapshot_date" not in frame.columns:
        raise KeyError("Exp126 requires canonical_project_id and snapshot_date")

    out = frame.copy()
    out["_exp126_row_order"] = np.arange(len(out), dtype=int)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    for column in QUALITY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    pieces: list[pd.DataFrame] = []
    for _, group0 in out.groupby("canonical_project_id", sort=False, dropna=False):
        group = group0.sort_values(["snapshot_date", "_exp126_row_order"], kind="mergesort").copy()
        dates = pd.to_datetime(group["snapshot_date"], errors="coerce")
        gap = dates.diff().dt.days.astype(float)
        duration = _numeric(group, "duration_ratio")
        revised = _numeric(group, "revised_cost_cr")
        cost = _numeric(group, "cost_escalation_percentage")
        slip = _numeric(group, "schedule_slippage_days")
        expenditure = _numeric(group, "expenditure_ratio")

        missing = group[QUALITY_COLUMNS].isna().sum(axis=1).astype(float)
        unchanged = pd.DataFrame(
            {
                "cost": cost.eq(cost.shift()),
                "slip": slip.eq(slip.shift()),
                "exp": expenditure.eq(expenditure.shift()),
                "revised": revised.eq(revised.shift()),
            },
            index=group.index,
        ).sum(axis=1).astype(float)

        group["is_report_gap_days"] = gap
        group["is_report_gap_mean3"] = gap.rolling(3, min_periods=1).mean()
        group["is_report_gap_std6"] = gap.rolling(6, min_periods=2).std()
        group["is_snapshot_index"] = np.arange(len(group), dtype=float)
        group["is_missing_count"] = missing
        group["is_missing_delta"] = missing.diff()
        group["is_unchanged_core_count"] = unchanged
        group["is_stale_report_flag"] = gap.gt(45).astype(float)
        group["is_gap_x_duration"] = gap * duration
        group["is_missing_x_duration"] = missing * duration
        pieces.append(group)

    if not pieces:
        return out.drop(columns=["_exp126_row_order"])

    result = pd.concat(pieces, axis=0)
    result = result.sort_values("_exp126_row_order", kind="mergesort").drop(columns=["_exp126_row_order"])
    return result


def fit_experiment(end: int, output: str):
    context = prepare_context(end)
    oof = production_oof(context)
    score = context["cohort"].copy()
    score["production_prediction"] = context["production_delay"]

    oof = add_reporting_behavior_features(oof)
    score = add_reporting_behavior_features(score)

    correction, details = fit_residual(
        oof,
        score,
        FEATURES,
        seed=12601,
    )
    details.update(
        {
            "ported_from_repository": "adityaab2007/SIH-26-Aditya",
            "ported_from_pr": 19,
            "source_experiment": "Exp126 Reporting Cadence and Data-Quality Behavior",
            "adaptation": "delay-only; current post-Exp113 production/OOF harness",
            "causal_as_of_features_only": True,
            "canonical_training_execution": "performance wrapper only; model logic unchanged",
        }
    )
    return persist(
        EXPERIMENT_ID,
        NAME,
        context,
        context["production_delay"] + correction,
        details,
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit_experiment(args.end, args.output)


if __name__ == "__main__":
    main()
