"""Experiment B: temporal schedule-credibility priors for Delay.

Government revised-completion dates are themselves forecasts.  This experiment
learns, from strictly earlier completed projects, how optimistic/pessimistic those
forecasts historically were for an agency/sector and exposes only the shrunk
credibility prior to the current Exp113 residual layer.
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

EXPERIMENT_ID = "exp_b"
EXPERIMENT_NAME = "B — official schedule credibility prior"
SEED = 13101
PRIOR_FEATURES = [
    "exp_b_schedule_bias_days",
    "exp_b_schedule_bias_support",
    "exp_b_adjusted_revised_remaining_days",
    "exp_b_bias_to_planned_duration",
    "exp_b_revised_remaining_days",
]
RESIDUAL_FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "exp34_schedule_revision_count",
    "exp58_delay_hier_prior",
    "exp58_group_support",
    *PRIOR_FEATURES,
]


def _norm(frame: pd.DataFrame, col: str) -> pd.Series:
    existing = "_norm_" + col
    if existing in frame:
        return frame[existing].astype("string").fillna("<NA>")
    return (
        frame.get(col, pd.Series("<NA>", index=frame.index))
        .astype("string")
        .fillna("<NA>")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
        .replace("", "<NA>")
    )


def _project_bias(reference: pd.DataFrame) -> pd.DataFrame:
    ref = reference.copy()
    ref["_agency"] = _norm(ref, "implementing_agency")
    ref["_sector"] = _norm(ref, "sector")
    revised = pd.to_datetime(ref.get("revised_completion_date"), errors="coerce")
    completion = pd.to_datetime(ref.get("completion_date"), errors="coerce")
    ref["_schedule_error_days"] = (completion - revised).dt.days.astype(float)
    valid = ref.dropna(subset=["canonical_project_id", "_schedule_error_days"]).copy()
    if valid.empty:
        raise ValueError("Experiment B requires historical revised and actual completion dates")
    projects = (
        valid.groupby("canonical_project_id", as_index=False)
        .agg(
            schedule_error_days=("_schedule_error_days", "median"),
            agency=("_agency", "last"),
            sector=("_sector", "last"),
        )
    )
    return projects


def _fit_priors(reference: pd.DataFrame, strength: float = 20.0) -> dict:
    projects = _project_bias(reference)
    global_bias = float(projects["schedule_error_days"].median())
    maps = {}
    for keys in [("agency", "sector"), ("sector",), ("agency",)]:
        grouped = projects.groupby(list(keys), dropna=False)["schedule_error_days"].agg(["median", "count"]).reset_index()
        grouped["value"] = (
            grouped["count"] * grouped["median"] + strength * global_bias
        ) / (grouped["count"] + strength)
        maps[keys] = grouped
    return {"global": global_bias, "maps": maps, "projects": len(projects), "strength": strength}


def attach_schedule_credibility(reference: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    priors = _fit_priors(reference)
    out = score.copy()
    out["_agency"] = _norm(out, "implementing_agency")
    out["_sector"] = _norm(out, "sector")
    values = np.full(len(out), float(priors["global"]), dtype=float)
    support = np.zeros(len(out), dtype=float)
    unresolved = np.ones(len(out), dtype=bool)

    for keys in [("agency", "sector"), ("sector",), ("agency",)]:
        lookup = priors["maps"][keys]
        left_cols = ["_" + key for key in keys]
        right = lookup.rename(columns={key: "_" + key for key in keys})
        merged = out[left_cols].merge(right, on=left_cols, how="left", sort=False)
        found = merged["value"].notna().to_numpy() & unresolved
        values[found] = merged.loc[found, "value"].to_numpy(float)
        support[found] = merged.loc[found, "count"].to_numpy(float)
        unresolved[found] = False

    snapshot = pd.to_datetime(out.get("snapshot_date"), errors="coerce")
    revised = pd.to_datetime(out.get("revised_completion_date"), errors="coerce")
    revised_remaining = (revised - snapshot).dt.days.astype(float)
    planned_duration = pd.to_numeric(out.get("planned_duration_days"), errors="coerce")

    out["exp_b_schedule_bias_days"] = values
    out["exp_b_schedule_bias_support"] = support
    out["exp_b_revised_remaining_days"] = revised_remaining
    out["exp_b_adjusted_revised_remaining_days"] = revised_remaining + values
    out["exp_b_bias_to_planned_duration"] = values / planned_duration.replace(0, np.nan)
    return out.drop(columns=["_agency", "_sector"], errors="ignore")


def _forward_prior_oof(oof: pd.DataFrame) -> pd.DataFrame:
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in year_col.dropna().unique())
    parts = []
    for year in years[1:]:
        reference = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if reference["canonical_project_id"].nunique() < 20 or validation.empty:
            continue
        parts.append(attach_schedule_credibility(reference, validation))
    if not parts:
        raise ValueError("Experiment B has no forward schedule-credibility folds")
    return pd.concat(parts, ignore_index=True)


def fit_experiment(training_end: int, output: str):
    ctx = prepare_context(training_end)
    oof = production_oof(ctx, max_folds=6)
    meta = _forward_prior_oof(oof)
    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_delay"]
    score = attach_schedule_credibility(oof, score)
    correction, details = fit_residual(meta, score, RESIDUAL_FEATURES, SEED)
    details.update(
        {
            "changed_dimension": "official_schedule_credibility",
            "one_project_contribution_to_prior": True,
            "prior_fit_for_meta": "strictly earlier completion/OOF years",
            "holdout_outcomes_used_for_prior": False,
            "prior_features": PRIOR_FEATURES,
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
