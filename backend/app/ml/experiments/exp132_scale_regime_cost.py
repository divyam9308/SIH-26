"""Exp132: scale-conditioned cost residual challenger; audit restricted to 2001-2021."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights, build_training_dataset
from backend.app.ml.monthly_training import _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp105_exp113_baseline import (
    train_window_with_promoted_cost_and_delay as train_production,
)

OOF_YEARS = (2018, 2019, 2020, 2021)


def metric(frame, prediction):
    return float(
        _regression_metrics(
            frame["actual_cost_overrun_percentage"],
            prediction,
            frame["sample_weight"],
            frame["canonical_project_id"],
        )["MAE"]
    )


def add_scale(frame, priors=None):
    work = frame.copy()
    cost = pd.to_numeric(work["approved_cost_cr"], errors="coerce").clip(lower=0)
    duration = pd.to_numeric(work["planned_duration_days"], errors="coerce").clip(lower=30)
    work["log_approved_cost"] = np.log1p(cost)
    work["capital_intensity_cr_per_day"] = cost / duration
    if priors is None:
        medians = work.assign(_cost=cost).groupby("sector")["_cost"].median().to_dict()
        global_median = float(cost.median())
    else:
        medians, global_median = priors
    denominator = work["sector"].map(medians).fillna(global_median).replace(
        0, global_median if global_median else 1
    )
    work["sector_relative_cost_ratio"] = cost / denominator
    return work, (medians, global_median)


def design(left, right, features):
    left_data = {}
    right_data = {}
    columns = []
    for column in features:
        if column not in left or column not in right:
            continue
        a = pd.to_numeric(left[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(right[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(a.median()) if a.notna().any() else 0.0
        columns.append(column)
        left_data[column] = a.fillna(median)
        right_data[column] = b.fillna(median)
    return columns, pd.DataFrame(left_data, index=left.index), pd.DataFrame(right_data, index=right.index)


def _training_context():
    data, identity = build_training_dataset()
    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, 2001, 2021, 2025)
    cohort = assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
    return {
        "data": data,
        "identity": identity,
        "train": train,
        "cohort": cohort,
    }


def _selected_oof_folds(train):
    completion_year = pd.to_numeric(train["completion_year"], errors="coerce")
    years = sorted(int(value) for value in completion_year.dropna().unique())[-4:]
    folds = {}
    for year in years:
        validation = train.loc[completion_year == year].copy()
        if year - 1 < 2005 or validation["canonical_project_id"].nunique() < 3:
            continue
        folds[int(year)] = validation
    return folds


def _strict_production_oof_part(ctx, validation, year):
    train_end = int(year) - 1
    with tempfile.TemporaryDirectory(prefix=f"exp132-{year}-") as td:
        root = Path(td) / "models"
        train_production(
            2001,
            train_end,
            int(year),
            data=ctx["data"],
            identity=ctx["identity"],
            artifact_root=root,
            verify_frozen_reference=False,
        )
        model = joblib.load(root / f"2001_{train_end}" / "cost_model.pkl")
        prediction = np.asarray(model.predict(validation), dtype=float)
    part = validation.copy()
    part["production_prediction"] = prediction
    part["residual"] = (
        pd.to_numeric(part["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
        - prediction
    )
    part["oof_year"] = int(year)
    return part


def build_oof_fold(year, output):
    ctx = _training_context()
    folds = _selected_oof_folds(ctx["train"])
    if int(year) not in folds:
        raise ValueError(f"OOF year {year} not selected; expected {sorted(folds)}")
    part = _strict_production_oof_part(ctx, folds[int(year)], int(year))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(part, path, compress=3)
    print(f"EXP132_PRODUCTION_OOF_FOLD_COMPLETED={year}; rows={len(part)}", flush=True)
    return path


def load_oof_dir(directory, expected=OOF_YEARS):
    paths = sorted(Path(directory).glob("cost-oof-*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No Exp132 Cost OOF artifacts in {directory}")
    parts = [joblib.load(path) for path in paths]
    years = [int(pd.to_numeric(part["oof_year"], errors="raise").iloc[0]) for part in parts]
    if tuple(sorted(years)) != tuple(expected):
        raise ValueError(f"OOF years {sorted(years)} != {list(expected)}")
    if len(set(years)) != len(years):
        raise ValueError("Duplicate Exp132 OOF year artifacts")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["oof_year", "canonical_project_id", "snapshot_date"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_precomputed_oof(oof, train):
    folds = _selected_oof_folds(train)
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


def fit_experiment(end=2021, output="reports/experiments/exp132_cost_2001_2021.json", precomputed_oof=None):
    if end != 2021:
        raise ValueError("Exp132 audit is restricted to 2001-2021")

    ctx = _training_context()
    train = ctx["train"]
    cohort = ctx["cohort"]

    with tempfile.TemporaryDirectory(prefix="exp132-prod-") as td:
        root = Path(td) / "models"
        train_production(
            2001,
            2021,
            2025,
            data=ctx["data"],
            identity=ctx["identity"],
            artifact_root=root,
        )
        model = joblib.load(root / "2001_2021" / "cost_model.pkl")
        production = np.asarray(model.predict(cohort), dtype=float)

    if precomputed_oof is None:
        folds = _selected_oof_folds(train)
        parts = [_strict_production_oof_part(ctx, folds[year], year) for year in OOF_YEARS]
        production_oof = pd.concat(parts, ignore_index=True)
    else:
        production_oof = _validate_precomputed_oof(precomputed_oof, train)

    production_oof, priors = add_scale(production_oof)
    score, _ = add_scale(cohort, priors)
    features = [
        "production_prediction",
        "cost_escalation_percentage",
        "duration_ratio",
        "log_approved_cost",
        "capital_intensity_cr_per_day",
        "sector_relative_cost_ratio",
    ]
    years = sorted(int(value) for value in production_oof["oof_year"].unique())
    meta = []
    for year in years[1:]:
        fitting = production_oof.loc[production_oof["oof_year"] < year]
        validation = production_oof.loc[production_oof["oof_year"] == year]
        _, x_fit, x_val = design(fitting, validation, features)
        residual_model = LGBMRegressor(
            objective="huber",
            alpha=0.9,
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=8,
            min_child_samples=50,
            random_state=132,
            verbosity=-1,
            n_jobs=1,
        )
        residual_model.fit(x_fit, fitting["residual"], sample_weight=fitting["sample_weight"])
        cap = max(float(np.nanquantile(np.abs(fitting["residual"]), 0.9)), 1e-9)
        ratio = np.clip(
            pd.to_numeric(validation["log_approved_cost"])
            / max(float(pd.to_numeric(fitting["log_approved_cost"]).median()), 1e-9),
            0.5,
            2.0,
        )
        correction = np.clip(
            residual_model.predict(x_val),
            -cap * (0.8 + 0.4 * ratio),
            cap * (0.8 + 0.4 * ratio),
        )
        meta.append((validation, correction))

    best = (1e99, 0.0)
    for scale in (0, 0.25, 0.5, 0.75, 1):
        candidate = float(
            np.mean(
                [
                    np.average(
                        np.abs(
                            validation["actual_cost_overrun_percentage"]
                            - (validation["production_prediction"] + scale * correction)
                        ),
                        weights=validation["sample_weight"],
                    )
                    for validation, correction in meta
                ]
            )
        )
        best = min(best, (candidate, float(scale)))

    _, x_fit, x_score = design(production_oof, score, features)
    residual_model = LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=8,
        min_child_samples=50,
        random_state=132,
        verbosity=-1,
        n_jobs=1,
    )
    residual_model.fit(x_fit, production_oof["residual"], sample_weight=production_oof["sample_weight"])
    cap = max(float(np.nanquantile(np.abs(production_oof["residual"]), 0.9)), 1e-9)
    ratio = np.clip(
        pd.to_numeric(score["log_approved_cost"])
        / max(float(pd.to_numeric(production_oof["log_approved_cost"]).median()), 1e-9),
        0.5,
        2.0,
    )
    correction = np.clip(
        residual_model.predict(x_score),
        -cap * (0.8 + 0.4 * ratio),
        cap * (0.8 + 0.4 * ratio),
    )
    prediction = production + best[1] * correction
    production_mae = metric(cohort, production)
    experiment_mae = metric(cohort, prediction)
    result = {
        "experiment_id": "exp132",
        "training_end": 2021,
        "production_cost_mae": production_mae,
        "experiment_cost_mae": experiment_mae,
        "cost_improvement_percentage": (production_mae - experiment_mae) / production_mae * 100,
        "selected_scale": best[1],
        "holdout_used_for_selection": False,
        "oof_years": list(OOF_YEARS),
        "scientific_verdict": "PROMOTION CANDIDATE" if experiment_mae < production_mae else "DO NOT PROMOTE",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result
