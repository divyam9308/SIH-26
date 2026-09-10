"""Exp126: Aditya reporting-cadence/data-quality delay challenger port.

Ports the successful delay-side idea from adityaab2007/SIH-26-Aditya PR #19
onto this repository's newer post-Exp113 leakage-safe comparison harness.
Only causal current/past snapshot behavior is engineered; production code and
cost predictions remain untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import backend.app.ml.experiments.post_exp113_delay_common as _common
from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.production_exp105_exp113_fast import (
    train_window_with_promoted_cost_and_delay as _fast_train_current_production,
)
from backend.app.ml.experiments.post_exp113_delay_common import (
    _production_oof_fold,
    fit_residual,
    forward_folds,
    persist,
    prepare_context,
    production_oof,
)
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors

# Branch-local execution substitution only: the fast wrapper calls the exact
# canonical Exp105+Exp113 trainer while parallelizing independent internal OOF
# work. This keeps the experiment/model/evaluation contract unchanged.
_common.train_current_production = _fast_train_current_production

EXPERIMENT_ID = "exp126"
NAME = "Reporting Cadence and Data-Quality Behavior (Aditya PR #19 port)"
OOF_YEARS = (2016, 2017, 2018, 2019, 2020, 2021)
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


def _training_context_2021():
    data, identity = build_training_dataset()
    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, 2001, 2021, 2025)
    train, _, _ = _build_temporal_delay_priors(train, test)
    return {"full_data": data, "identity": identity, "train": train}


def _selected_oof_folds(train, max_folds=6):
    return {
        int(year): validation
        for _, validation, year in forward_folds(train, max_folds)
        if int(year) - 1 >= 2005
    }


def build_oof_fold(year: int, output: str):
    """Build one exact strict-forward production OOF fold for CI sharding."""
    ctx = _training_context_2021()
    folds = _selected_oof_folds(ctx["train"], max_folds=6)
    if int(year) not in folds:
        raise ValueError(f"OOF year {year} not selected; expected {sorted(folds)}")
    part = _production_oof_fold(
        folds[int(year)], int(year), ctx["full_data"], ctx["identity"]
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(part, path, compress=3)
    print(f"EXP126_PRODUCTION_OOF_FOLD_COMPLETED={year}; rows={len(part)}", flush=True)
    return path


def load_oof_dir(directory: str | Path, expected=OOF_YEARS):
    paths = sorted(Path(directory).glob("delay-oof-*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No Exp126 Delay OOF artifacts in {directory}")
    parts = [joblib.load(path) for path in paths]
    years = [int(pd.to_numeric(part["oof_year"], errors="raise").iloc[0]) for part in parts]
    if tuple(sorted(years)) != tuple(expected):
        raise ValueError(f"OOF years {sorted(years)} != {list(expected)}")
    if len(set(years)) != len(years):
        raise ValueError("Duplicate Exp126 OOF year artifacts")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["oof_year", "canonical_project_id", "snapshot_date"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_precomputed_oof(oof: pd.DataFrame, train: pd.DataFrame):
    folds = _selected_oof_folds(train, max_folds=6)
    years = sorted(int(value) for value in pd.to_numeric(oof["oof_year"], errors="raise").unique())
    if tuple(years) != tuple(OOF_YEARS):
        raise ValueError(f"Precomputed OOF years {years} != {list(OOF_YEARS)}")
    if tuple(sorted(folds)) != tuple(OOF_YEARS):
        raise ValueError(f"Current training frame selected OOF years {sorted(folds)} != {list(OOF_YEARS)}")
    for year in OOF_YEARS:
        expected = folds[year]
        actual = oof.loc[pd.to_numeric(oof["oof_year"], errors="coerce") == year]
        if len(actual) != len(expected):
            raise ValueError(f"OOF {year} row count {len(actual)} != expected {len(expected)}")
        expected_keys = set(
            zip(
                expected["canonical_project_id"].astype(str),
                pd.to_datetime(expected["snapshot_date"], errors="coerce").astype(str),
            )
        )
        actual_keys = set(
            zip(
                actual["canonical_project_id"].astype(str),
                pd.to_datetime(actual["snapshot_date"], errors="coerce").astype(str),
            )
        )
        if actual_keys != expected_keys:
            raise ValueError(f"OOF {year} row identity mismatch")
    return oof.copy()


def fit_experiment(end: int, output: str, precomputed_oof: pd.DataFrame | None = None):
    context = prepare_context(end)
    if precomputed_oof is None:
        oof = production_oof(context)
    else:
        if end != 2021:
            raise ValueError("Precomputed Exp126 OOF shards are defined only for the 2001-2021 audit")
        oof = _validate_precomputed_oof(precomputed_oof, context["train"])
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
            "oof_years": list(OOF_YEARS) if precomputed_oof is not None else None,
            "oof_execution": "precomputed strict-forward shards" if precomputed_oof is not None else "in-process",
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
