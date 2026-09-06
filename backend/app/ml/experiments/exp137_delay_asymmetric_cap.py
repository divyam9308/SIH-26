"""Exp137: asymmetric Delay cap expansion for tail recovery.

Strict-forward challenger on top of the current Exp113 production Delay stack.
The production anchor is never refit on holdout labels; scale selection uses only
forward OOF folds. The experiment is intentionally restricted to 2001-2022 ->
2023-2025.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.post_exp113_delay_common import (
    numeric_design,
    prepare_context,
    production_oof,
)
from backend.app.ml.monthly_training import _json_safe, _regression_metrics

LOWER_CAP_DAYS = -500.0
UPPER_CAP_DAYS = 1500.0
FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "expenditure_ratio",
    "cost_escalation_percentage",
    "physical_progress",
    "cost_growth_velocity_6m",
    "elapsed_duration_days",
    "is_railways_sector",
    "elapsed_over_10yr",
    "stagnant_progress_24m",
]


def add_structural_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    sector = x.get("sector", pd.Series("", index=x.index)).astype("string").fillna("")
    elapsed = pd.to_numeric(x.get("elapsed_duration_days"), errors="coerce")
    velocity = pd.to_numeric(x.get("cost_growth_velocity_6m"), errors="coerce")
    duration_ratio = pd.to_numeric(x.get("duration_ratio"), errors="coerce")
    x["is_railways_sector"] = sector.str.contains("rail", case=False, regex=False).astype(float)
    x["elapsed_over_10yr"] = (elapsed > 3650.0).fillna(False).astype(float)
    x["stagnant_progress_24m"] = ((velocity <= 0.0) & (duration_ratio > 1.5)).fillna(False).astype(float)
    return x


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return _regression_metrics(
        frame["actual_delay_days"],
        np.asarray(prediction, dtype=float),
        frame["sample_weight"],
        frame["canonical_project_id"],
    )


def _stage_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    if "lifecycle_stage" not in frame.columns:
        return {}
    pred = np.asarray(prediction, dtype=float)
    result = {}
    stage_col = frame["lifecycle_stage"].astype("string").str.lower()
    for stage in ("early", "mid", "late", "very_late"):
        mask = stage_col.eq(stage)
        if int(mask.sum()) < 2:
            result[stage] = {"available": False}
            continue
        sub = frame.loc[mask]
        m = _regression_metrics(
            sub["actual_delay_days"], pred[mask.to_numpy()], sub["sample_weight"], sub["canonical_project_id"]
        )
        result[stage] = {"available": True, **m}
    return result


def _fit_asymmetric_residual(oof: pd.DataFrame, score: pd.DataFrame):
    oof = add_structural_lag_features(oof)
    score = add_structural_lag_features(score)
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in year_col.dropna().unique())
    meta = []
    for year in years[1:]:
        fitting = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(fitting) < 80 or validation.empty:
            continue
        cols, _, x_fit, x_val = numeric_design(fitting, validation, FEATURES)
        model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=8,
            min_child_samples=60,
            reg_alpha=5,
            reg_lambda=25,
            random_state=137,
            verbosity=-1,
            n_jobs=1,
        )
        residual = pd.to_numeric(fitting["residual"], errors="coerce").fillna(0.0).to_numpy(float)
        weight = pd.to_numeric(fitting["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
        model.fit(x_fit, residual, sample_weight=weight)
        correction = np.clip(np.asarray(model.predict(x_val), dtype=float), LOWER_CAP_DAYS, UPPER_CAP_DAYS)
        meta.append((validation, correction))
    if not meta:
        raise ValueError("Exp137 requires forward meta-OOF predictions")

    best = (float("inf"), 0.0)
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        values, weights = [], []
        for validation, correction in meta:
            actual = pd.to_numeric(validation["actual_delay_days"], errors="coerce").to_numpy(float)
            anchor = pd.to_numeric(validation["production_prediction"], errors="coerce").to_numpy(float)
            w = pd.to_numeric(validation["sample_weight"], errors="coerce").to_numpy(float)
            pred = np.maximum(0.0, anchor + scale * correction)
            values.append(float(np.average(np.abs(actual - pred), weights=w)))
            weights.append(max(float(np.nansum(w)), 1e-9))
        candidate = (float(np.average(values, weights=weights)), float(scale))
        if candidate < best:
            best = candidate

    cols, medians, x_fit, x_score = numeric_design(oof, score, FEATURES)
    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=8,
        min_child_samples=60,
        reg_alpha=5,
        reg_lambda=25,
        random_state=137,
        verbosity=-1,
        n_jobs=1,
    )
    residual = pd.to_numeric(oof["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    model.fit(x_fit, residual, sample_weight=weight)
    raw = np.asarray(model.predict(x_score), dtype=float)
    correction = best[1] * np.clip(raw, LOWER_CAP_DAYS, UPPER_CAP_DAYS)
    return correction, {
        "selected_scale": best[1],
        "features": cols,
        "medians": medians,
        "lower_cap_days": LOWER_CAP_DAYS,
        "upper_cap_days": UPPER_CAP_DAYS,
        "meta_oof_years": years[1:],
    }


def fit_experiment(end: int = 2022, output: str = "reports/experiments/exp137_delay_asymmetric_cap_2001_2022.json") -> dict:
    if end != 2022:
        raise ValueError("Exp137 canonical audit is restricted to 2001-2022 -> 2023-2025")
    ctx = prepare_context(end)
    oof = production_oof(ctx, max_folds=6)
    score = ctx["cohort"].copy()
    score["production_prediction"] = np.asarray(ctx["production_delay"], dtype=float)
    correction, details = _fit_asymmetric_residual(oof, score)
    production = np.asarray(ctx["production_delay"], dtype=float)
    experiment = np.maximum(0.0, production + correction)
    production_metrics = _metrics(score, production)
    experiment_metrics = _metrics(score, experiment)
    production_stage = _stage_metrics(score, production)
    experiment_stage = _stage_metrics(score, experiment)
    success = (
        float(experiment_metrics["MAE"]) <= float(production_metrics["MAE"])
        and float(experiment_metrics["RMSE"]) <= float(production_metrics["RMSE"])
        and float(experiment_metrics["R2"]) > float(production_metrics["R2"])
    )
    result = {
        "experiment_id": "exp137_delay_asymmetric_cap",
        "experiment_name": "Asymmetric Delay Cap Expansion for Tail Recovery",
        "scope": "delay",
        "training_start": 2001,
        "training_end": 2022,
        "test_start": 2023,
        "test_end": 2025,
        "production_delay_metrics": production_metrics,
        "experiment_delay_metrics": experiment_metrics,
        "production_stage_metrics": production_stage,
        "experiment_stage_metrics": experiment_stage,
        "mae_delta_days": round(float(experiment_metrics["MAE"]) - float(production_metrics["MAE"]), 3),
        "rmse_delta_days": round(float(experiment_metrics["RMSE"]) - float(production_metrics["RMSE"]), 3),
        "r2_delta": round(float(experiment_metrics["R2"]) - float(production_metrics["R2"]), 4),
        "holdout_used_for_selection": False,
        "promotion_allowed": False,
        "scientific_verdict": "PROMOTION CANDIDATE" if success else "DO NOT PROMOTE",
        "details": details,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result
