"""Exp143: Recency-Drift Weighted OOF Cost Correction.

Challenger model on top of the current Exp105 Cost production stack.
Applies exponential recency weighting to historical OOF evidence during residual
booster training to adapt dynamically to modern inflation and execution regimes.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml import production_exp105_exp113_baseline as prod
from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import (
    _json_safe,
    _regression_metrics,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp35_baseline import (
    CALIBRATION_GATE_FEATURE,
    _aft_routing_limit,
    _select_aft_calibration_projects,
)
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors

EXPERIMENT_ID = "exp_143"
EXPERIMENT_NAME = "Recency-Drift Weighted OOF Cost Correction"
COST_MODEL_ID = "exp61_plus_exp143_recency_drift_cost_v1"

EXP143_FEATURES = [
    "production_prediction",
    "approved_cost_cr",
    "cost_escalation_percentage",
    "expenditure_ratio",
    "duration_ratio",
    "progress_deviation",
    "physical_progress",
    "cost_growth_velocity_6m",
]


def calculate_recency_weights(
    years: pd.Series,
    base_weights: np.ndarray,
    decay_lambda: float = 1.2,
) -> np.ndarray:
    """Apply exponential recency weighting w_t = w * exp(lambda * (t - min_t)/(max_t - min_t))."""
    y = pd.to_numeric(years, errors="coerce").fillna(2005).to_numpy(float)
    min_y = float(np.min(y))
    max_y = float(np.max(y))
    span = max(max_y - min_y, 1.0)
    norm_t = (y - min_y) / span
    recency_factor = np.exp(decay_lambda * norm_t)
    combined = np.asarray(base_weights, dtype=float) * recency_factor
    return combined / np.mean(combined)


def _stage_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    if "lifecycle_stage" not in frame.columns:
        return {}
    pred = np.asarray(prediction, dtype=float)
    stage_col = frame["lifecycle_stage"].astype("string").str.lower()
    result = {}
    for stage in ("early", "mid", "late", "very_late"):
        mask = stage_col.eq(stage)
        if int(mask.sum()) < 2:
            result[stage] = {"available": False}
            continue
        sub = frame.loc[mask]
        result[stage] = {
            "available": True,
            **_regression_metrics(
                sub["actual_cost_overrun_percentage"],
                pred[mask.fillna(False).to_numpy(dtype=bool)],
                sub["sample_weight"],
                sub["canonical_project_id"],
            ),
        }
    return result


def _fit_recency_drift_cost_residual_layer(
    oof: pd.DataFrame,
    score: pd.DataFrame,
    *,
    seed: int = 143,
    min_fit_rows: int = 80,
    meta_estimators: int = 100,
    final_estimators: int = 140,
    decay_lambda: float = 1.2,
):
    work = oof.copy()
    actual = pd.to_numeric(work["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(work["production_prediction"], errors="coerce").to_numpy(float)
    work["residual"] = actual - anchor

    years = sorted(
        int(x)
        for x in pd.to_numeric(work["oof_year"], errors="coerce").dropna().unique()
    )
    meta_predictions: list[tuple[pd.DataFrame, np.ndarray]] = []
    for year in years[1:]:
        year_col = pd.to_numeric(work["oof_year"], errors="coerce")
        fitting = work.loc[year_col < year].copy()
        validation = work.loc[year_col == year].copy()
        if len(fitting) < min_fit_rows or validation.empty:
            continue

        _, _, x_fit, x_val = prod._numeric_design(fitting, validation, EXP143_FEATURES)
        model = LGBMRegressor(
            n_estimators=meta_estimators,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=8,
            min_child_samples=50,
            reg_alpha=4.0,
            reg_lambda=20.0,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )
        residual = pd.to_numeric(fitting["residual"], errors="coerce").fillna(0.0).to_numpy(float)
        raw_weight = pd.to_numeric(fitting["sample_weight"], errors="coerce").fillna(1.0).to_numpy(float)
        recency_weight = calculate_recency_weights(fitting["oof_year"], raw_weight, decay_lambda=decay_lambda)

        model.fit(x_fit, residual, sample_weight=recency_weight)
        cap = max(prod._weighted_quantile(np.abs(residual), raw_weight, 0.90), 1e-9)
        correction = np.clip(np.asarray(model.predict(x_val), dtype=float), -cap, cap)
        meta_predictions.append((validation, correction))

    if not meta_predictions:
        raise ValueError("Exp143 has no forward meta-OOF predictions")

    best = (float("inf"), 0.0)
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        fold_mae: list[float] = []
        fold_weight: list[float] = []
        for validation, correction in meta_predictions:
            anchor_raw = pd.to_numeric(
                validation["production_prediction"], errors="coerce"
            ).to_numpy(float)
            prediction = anchor_raw + scale * correction
            actual_raw = pd.to_numeric(
                validation["actual_cost_overrun_percentage"], errors="coerce"
            ).to_numpy(float)
            weight = pd.to_numeric(
                validation["sample_weight"], errors="coerce"
            ).to_numpy(float)
            fold_mae.append(prod._weighted_mae(actual_raw, prediction, weight))
            fold_weight.append(max(float(np.nansum(weight)), 1e-9))
        candidate = (float(np.average(fold_mae, weights=fold_weight)), float(scale))
        if candidate < best:
            best = candidate

    selected_scale = float(best[1])

    cols, medians, x_fit, x_score = prod._numeric_design(work, score, EXP143_FEATURES)
    final_model = LGBMRegressor(
        n_estimators=final_estimators,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=8,
        min_child_samples=50,
        reg_alpha=4.0,
        reg_lambda=20.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    residual = pd.to_numeric(work["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    raw_weight = pd.to_numeric(work["sample_weight"], errors="coerce").fillna(1.0).to_numpy(float)
    final_recency_weight = calculate_recency_weights(work["oof_year"], raw_weight, decay_lambda=decay_lambda)

    final_model.fit(x_fit, residual, sample_weight=final_recency_weight)
    cap = max(prod._weighted_quantile(np.abs(residual), raw_weight, 0.90), 1e-9)
    raw = np.clip(np.asarray(final_model.predict(x_score), dtype=float), -cap, cap)
    correction = selected_scale * raw

    return correction, {
        "selected_scale": selected_scale,
        "features": cols,
        "medians": medians,
        "decay_lambda": decay_lambda,
        "correction_cap": float(cap),
        "meta_oof_years": years[1:],
    }


def train_window_with_exp143(
    training_start: int,
    training_end: int,
    test_end: int,
    *,
    data: pd.DataFrame,
    identity: pd.DataFrame | None = None,
    artifact_root: Path,
) -> dict:
    root = Path(artifact_root)
    result = prod.train_window_with_promoted_cost_and_delay(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=root,
        verify_frozen_reference=False,
    )
    target = root / f"{training_start}_{training_end}"
    current_cost_model = joblib.load(target / "cost_model.pkl")
    anchor_model = current_cost_model.base_model if hasattr(current_cost_model, "base_model") else current_cost_model

    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, training_start, training_end, test_end)
    prior_train, prior_test, _ = _build_temporal_delay_priors(train, test)
    cohort = _production_cost_evaluation_rows(prior_test).copy()
    calibration_ids = _select_aft_calibration_projects(
        cohort,
        limit=_aft_routing_limit(training_start, training_end, test_end),
    )
    cohort[CALIBRATION_GATE_FEATURE] = cohort["canonical_project_id"].astype("string").isin(calibration_ids)
    cohort = assign_project_balanced_weights(cohort)

    oof = prod._current_cost_oof(prior_train, anchor_model)
    anchor_prediction = np.asarray(anchor_model.predict(cohort), dtype=float)
    production_prediction = np.asarray(current_cost_model.predict(cohort), dtype=float)

    cohort_work = cohort.copy()
    cohort_work["production_prediction"] = anchor_prediction

    correction, details = _fit_recency_drift_cost_residual_layer(oof, cohort_work)
    exp143_prediction = anchor_prediction + correction

    prod_metrics = _regression_metrics(
        cohort["actual_cost_overrun_percentage"],
        production_prediction,
        cohort["sample_weight"],
        cohort["canonical_project_id"],
    )
    exp_metrics = _regression_metrics(
        cohort["actual_cost_overrun_percentage"],
        exp143_prediction,
        cohort["sample_weight"],
        cohort["canonical_project_id"],
    )
    prod_stage = _stage_metrics(cohort, production_prediction)
    exp_stage = _stage_metrics(cohort, exp143_prediction)

    success = (
        float(exp_metrics["MAE"]) <= float(prod_metrics["MAE"])
        and float(exp_metrics["RMSE"]) <= float(prod_metrics["RMSE"])
        and float(exp_metrics["R2"]) >= float(prod_metrics["R2"])
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "scope": "cost",
        "training_start": int(training_start),
        "training_end": int(training_end),
        "test_start": int(training_end) + 1,
        "test_end": int(test_end),
        "production_cost_metrics": prod_metrics,
        "experiment_cost_metrics": exp_metrics,
        "production_stage_metrics": prod_stage,
        "experiment_stage_metrics": exp_stage,
        "mae_delta": round(float(exp_metrics["MAE"]) - float(prod_metrics["MAE"]), 3),
        "rmse_delta": round(float(exp_metrics["RMSE"]) - float(prod_metrics["RMSE"]), 3),
        "r2_delta": round(float(exp_metrics["R2"]) - float(prod_metrics["R2"]), 4),
        "holdout_used_for_selection": False,
        "promotion_allowed": False,
        "scientific_verdict": "PROMOTION CANDIDATE" if success else "DO NOT PROMOTE",
        "details": details,
    }
    return {"production": result, "exp143": payload}
