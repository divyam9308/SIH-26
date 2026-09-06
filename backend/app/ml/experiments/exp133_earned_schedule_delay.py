"""Exp133: earned-schedule/calendar-velocity residual challenger."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
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

OOF_YEARS = (2016, 2017, 2018, 2019, 2020, 2021)


def earned(frame):
    work = frame.copy()
    planned = pd.to_numeric(work.get("planned_duration_days"), errors="coerce")
    progress = pd.to_numeric(work.get("physical_progress"), errors="coerce").clip(0, 100)
    elapsed = pd.to_numeric(work.get("elapsed_duration_days"), errors="coerce")
    if elapsed.isna().all() and "snapshot_date" in work and "start_date" in work:
        elapsed = (
            pd.to_datetime(work["snapshot_date"], errors="coerce")
            - pd.to_datetime(work["start_date"], errors="coerce")
        ).dt.days
    elapsed = elapsed.clip(lower=0)
    earned_schedule = planned * progress / 100.0
    velocity = earned_schedule / elapsed.clip(lower=30)
    velocity = velocity.replace([np.inf, -np.inf], np.nan)

    # Conservative prior of planned velocity 1.0; elapsed-time shrinkage prevents early-stage explosions.
    shrinkage_days = 180.0
    velocity_shrunk = (
        elapsed.fillna(0) * velocity.fillna(1.0) + shrinkage_days
    ) / (elapsed.fillna(0) + shrinkage_days)
    velocity_shrunk = velocity_shrunk.clip(lower=0.05, upper=2.0)
    remaining = (planned - earned_schedule).clip(lower=0) / velocity_shrunk
    projected_delay = (elapsed + remaining - planned).clip(lower=0)
    production = pd.to_numeric(work.get("production_prediction"), errors="coerce").fillna(0)

    work["es_velocity_shrunk"] = velocity_shrunk.fillna(1.0)
    work["es_projected_delay_days"] = projected_delay.fillna(production)
    work["es_divergence_gap"] = (
        work["es_projected_delay_days"] - production
    ).clip(-1000, 5000).fillna(0)
    return work


def _training_context():
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


def build_oof_fold(year, output):
    ctx = _training_context()
    folds = _selected_oof_folds(ctx["train"], max_folds=6)
    if int(year) not in folds:
        raise ValueError(f"OOF year {year} not selected; expected {sorted(folds)}")
    part = _production_oof_fold(
        folds[int(year)], int(year), ctx["full_data"], ctx["identity"]
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(part, path, compress=3)
    print(f"EXP133_PRODUCTION_OOF_FOLD_COMPLETED={year}; rows={len(part)}", flush=True)
    return path


def load_oof_dir(directory, expected=OOF_YEARS):
    paths = sorted(Path(directory).glob("delay-oof-*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No Exp133 Delay OOF artifacts in {directory}")
    parts = [joblib.load(path) for path in paths]
    years = [int(pd.to_numeric(part["oof_year"], errors="raise").iloc[0]) for part in parts]
    if tuple(sorted(years)) != tuple(expected):
        raise ValueError(f"OOF years {sorted(years)} != {list(expected)}")
    if len(set(years)) != len(years):
        raise ValueError("Duplicate Exp133 OOF year artifacts")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["oof_year", "canonical_project_id", "snapshot_date"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_precomputed_oof(oof, train):
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


def fit_experiment(
    end=2021,
    output="reports/experiments/exp133_delay_2001_2021.json",
    precomputed_oof=None,
):
    if end != 2021:
        raise ValueError("Exp133 audit is restricted to 2001-2021")

    ctx = prepare_context(end)
    if precomputed_oof is None:
        oof = production_oof(ctx, max_folds=6)
    else:
        oof = _validate_precomputed_oof(precomputed_oof, ctx["train"])
    oof = earned(oof)

    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_delay"]
    score = earned(score)
    features = [
        "production_prediction",
        "es_divergence_gap",
        "es_velocity_shrunk",
        "es_projected_delay_days",
        "duration_ratio",
        "schedule_slippage_days",
        "expenditure_ratio",
        "cost_escalation_percentage",
    ]
    correction, details = fit_residual(oof, score, features, 133)
    gap = pd.to_numeric(score["es_divergence_gap"], errors="coerce").fillna(0).to_numpy(float)
    base = float(details["cap"])
    upper = np.minimum(3000.0, base + 0.5 * np.maximum(gap, 0))
    correction = np.minimum(correction, upper)
    prediction = np.maximum(0, ctx["production_delay"] + correction)
    details.update(
        {
            "earned_schedule": "elapsed-time-shrunk calendar velocity",
            "adaptive_upper_cap": True,
            "oof_years": list(OOF_YEARS),
        }
    )
    return persist(
        "exp133",
        "Earned Schedule Velocity & Calendar Extension Hybrid",
        ctx,
        prediction,
        details,
        output,
    )
