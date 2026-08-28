"""Experiment 25: delay-only milestone trajectory challenger.

This experiment intentionally retains the promoted production cost model and
changes only delay. Milestone values come from the current official PAIMANA
snapshot (for example ``6/14``). Trajectory features are engineered on the full
monthly history using only the current and earlier rows for the same canonical
project, then joined onto the supervised snapshot frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import TRAJECTORIES

EXPERIMENT_ID = "exp_25"
EXPERIMENT_NAME = "Delay-only milestone trajectory"
EXPERIMENT_SCOPE = "delay"
MILESTONE_FEATURES = [
    "exp25_milestones_achieved",
    "exp25_milestones_total",
    "exp25_milestone_ratio",
    "exp25_milestones_remaining",
    "exp25_milestone_velocity",
    "exp25_milestone_delta",
    "exp25_milestone_stagnant",
    "exp25_months_since_milestone_change",
]


def add_milestone_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineer current/past-only milestone features on a monthly history frame."""
    result = frame.copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce")
    result["canonical_project_id"] = result["canonical_project_id"].astype("string")
    status = result.get("milestone_status", pd.Series(None, index=result.index)).astype("string")
    parts = status.str.extract(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
    result["exp25_milestones_achieved"] = pd.to_numeric(parts["done"], errors="coerce")
    result["exp25_milestones_total"] = pd.to_numeric(parts["total"], errors="coerce")
    done = result["exp25_milestones_achieved"]
    total = result["exp25_milestones_total"]
    result["exp25_milestone_ratio"] = (done / total).where(total.gt(0)).clip(0, 1)
    result["exp25_milestones_remaining"] = (total - done).where(total.notna() & done.notna()).clip(lower=0)
    for name in MILESTONE_FEATURES[4:]:
        result[name] = np.nan

    ordered = result.sort_values(["canonical_project_id", "snapshot_date"])
    for _, group in ordered.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group["snapshot_date"]
        achieved = group["exp25_milestones_achieved"]
        ratios = group["exp25_milestone_ratio"]
        months = dates.diff().dt.days / 30.4375
        delta = achieved.diff()
        velocity = ratios.diff().div(months).where(months.gt(0))
        stagnant = pd.Series(np.where(delta.notna(), (delta <= 0).astype(float), np.nan), index=idx)
        since = pd.Series(np.nan, index=idx, dtype=float)
        previous = np.nan
        last_change = None
        for row_index in idx:
            current = result.at[row_index, "exp25_milestones_achieved"]
            current_date = result.at[row_index, "snapshot_date"]
            if pd.isna(current) or pd.isna(current_date):
                continue
            if pd.isna(previous) or current != previous:
                last_change = current_date
            if last_change is not None:
                since.at[row_index] = max(0.0, (current_date - last_change).days / 30.4375)
            previous = current
        result.loc[idx, "exp25_milestone_velocity"] = velocity.to_numpy()
        result.loc[idx, "exp25_milestone_delta"] = delta.to_numpy()
        result.loc[idx, "exp25_milestone_stagnant"] = stagnant.to_numpy()
        result.loc[idx, "exp25_months_since_milestone_change"] = since.to_numpy()
    return result


def enrich_with_monthly_milestones(frame: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join causally engineered full-monthly milestone features to snapshots."""
    supervised = frame.copy()
    supervised["snapshot_date"] = pd.to_datetime(supervised["snapshot_date"], errors="coerce")
    supervised["canonical_project_id"] = supervised["canonical_project_id"].astype("string")
    if history is None:
        if TRAJECTORIES.exists():
            history = pd.read_csv(
                TRAJECTORIES,
                dtype={"canonical_project_id": "string"},
                low_memory=False,
            )
        else:
            history = supervised.copy()
    monthly = add_milestone_features(history)
    lookup = monthly[["canonical_project_id", "snapshot_date", *MILESTONE_FEATURES]].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )
    supervised = supervised.drop(columns=[c for c in MILESTONE_FEATURES if c in supervised], errors="ignore")
    return supervised.merge(
        lookup,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )


def decision(delay_improvement: float) -> str:
    """Cost is retained exactly, so promotion requires a strict Delay MAE gain."""
    return "PROMOTION CANDIDATE" if delay_improvement > 0 else "REGRESSION / DO NOT PROMOTE"
