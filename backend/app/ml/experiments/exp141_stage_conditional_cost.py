"""Exp141: Lifecycle-Stage Conditional Cost Meta-Learner.

Challenger model on top of the current Exp105 Cost production stack.
Decomposes residual correction into 4 stage-specialized models (early, mid, late, very_late),
optimizing scale calibration independently per lifecycle regime.
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

EXPERIMENT_ID = "exp_141"
EXPERIMENT_NAME = "Lifecycle-Stage Conditional Cost Meta-Learner"
COST_MODEL_ID = "exp61_plus_exp141_stage_conditional_cost_v1"

STAGES = ("early", "mid", "late", "very_late")

STAGE_FEATURES = {
    "early": [
        "production_prediction",
        "approved_cost_cr",
        "cost_escalation_percentage",
        "expenditure_ratio",
        "duration_ratio",
    ],
    "mid": [
        "production_prediction",
        "expenditure_ratio",
        "progress_deviation",
        "cost_growth_velocity_6m",
        "duration_ratio",
        "cost_escalation_percentage",
    ],
    "late": [
        "production_prediction",
        "schedule_slippage_days",
        "physical_progress",
        "progress_deviation",
        "cost_escalation_percentage",
        "expenditure_ratio",
    ],
    "very_late": [
        "production_prediction",
        "physical_progress",
        "progress_deviation",
        "cost_escalation_percentage",
        "expenditure_ratio",
    ],
}


def _normalize_stage(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().fillna("mid")


def _stage_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    if "lifecycle_stage" not in frame.columns:
        return {}
    pred = np.asarray(prediction, dtype=float)
    stage_col = _normalize_stage(frame["lifecycle_stage"])
    result = {}
    for stage in STAGES:
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


def _fit_stage_conditional_residual_layer(
    oof: pd.DataFrame,
    score: pd.DataFrame,
    *,
    seed: int = 141,
    min_stage_rows: int = 30,
    meta_estimators: int = 80,
    final_estimators: int = 120,
):
    work = oof.copy()
    actual = pd.to_numeric(work["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(work["production_prediction"], errors="coerce").to_numpy(float)
    work["residual"] = actual - anchor
    work["stage_norm"] = _normalize_stage(work.get("lifecycle_stage", pd.Series("mid", index=work.index)))

    years = sorted(
        int(x)
        for x in pd.to_numeric(work["oof_year"], errors="coerce").dropna().unique()
    )

    stage_scales: dict[str, float] = {}
    stage_models: dict[str, LGBMRegressor] = {}
    stage_cols: dict[str, list[str]] = {}
    stage_medians: dict[str, dict[str, float]] = {}
    stage_caps: dict[str, float] = {}

    for stage in STAGES:
        feats = STAGE_FEATURES.get(stage, STAGE_FEATURES["mid"])
        meta_predictions: list[tuple[pd.DataFrame, np.ndarray]] = []

        for year in years[1:]:
            year_col = pd.to_numeric(work["oof_year"], errors="coerce")
            fitting = work.loc[(year_col < year) & (work["stage_norm"] == stage)].copy()
            validation = work.loc[(year_col == year) & (work["stage_norm"] == stage)].copy()
            if len(fitting) < min_stage_rows or validation.empty:
                continue

            cols, _, x_fit, x_val = prod._numeric_design(fitting, validation, feats)
            model = LGBMRegressor(
                n_estimators=meta_estimators,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=8,
                min_child_samples=25,
                reg_alpha=3.0,
                reg_lambda=15.0,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            )
            residual = pd.to_numeric(fitting["residual"], errors="coerce").fillna(0.0).to_numpy(float)
            weight = pd.to_numeric(fitting["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
            model.fit(x_fit, residual, sample_weight=weight)
            cap = max(prod._weighted_quantile(np.abs(residual), weight, 0.90), 1e-9)
            correction = np.clip(np.asarray(model.predict(x_val), dtype=float), -cap, cap)
            meta_predictions.append((validation, correction))

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
            candidate = (
                float(np.average(fold_mae, weights=fold_weight)) if fold_mae else 0.0,
                float(scale),
            )
            if candidate < best:
                best = candidate

        selected_scale = float(best[1])
        stage_scales[stage] = selected_scale

        stage_work = work.loc[work["stage_norm"] == stage].copy()
        if len(stage_work) < min_stage_rows:
            stage_work = work.copy()

        cols, medians, x_fit, _ = prod._numeric_design(stage_work, stage_work, feats)
        stage_model = LGBMRegressor(
            n_estimators=final_estimators,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=8,
            min_child_samples=25,
            reg_alpha=3.0,
            reg_lambda=15.0,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )
        residual = pd.to_numeric(stage_work["residual"], errors="coerce").fillna(0.0).to_numpy(float)
        weight = pd.to_numeric(stage_work["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
        stage_model.fit(x_fit, residual, sample_weight=weight)
        cap = max(prod._weighted_quantile(np.abs(residual), weight, 0.90), 1e-9)

        stage_models[stage] = stage_model
        stage_cols[stage] = cols
        stage_medians[stage] = medians
        stage_caps[stage] = float(cap)

    score_norm = _normalize_stage(score.get("lifecycle_stage", pd.Series("mid", index=score.index)))
    total_correction = np.zeros(len(score), dtype=float)

    for stage in STAGES:
        mask = score_norm.eq(stage).to_numpy(dtype=bool)
        if not mask.any():
            continue
        sub_score = score.loc[mask].copy()
        x_score = prod._design_from_frozen(sub_score, stage_cols[stage], stage_medians[stage])
        raw_pred = np.clip(
            np.asarray(stage_models[stage].predict(x_score), dtype=float),
            -stage_caps[stage],
            stage_caps[stage],
        )
        total_correction[mask] = stage_scales[stage] * raw_pred

    return total_correction, {
        "stage_scales": stage_scales,
        "stage_caps": stage_caps,
        "stage_features": STAGE_FEATURES,
        "meta_oof_years": years[1:],
    }


def train_window_with_exp141(
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

    correction, details = _fit_stage_conditional_residual_layer(oof, cohort_work)
    exp141_prediction = anchor_prediction + correction

    prod_metrics = _regression_metrics(
        cohort["actual_cost_overrun_percentage"],
        production_prediction,
        cohort["sample_weight"],
        cohort["canonical_project_id"],
    )
    exp_metrics = _regression_metrics(
        cohort["actual_cost_overrun_percentage"],
        exp141_prediction,
        cohort["sample_weight"],
        cohort["canonical_project_id"],
    )
    prod_stage = _stage_metrics(cohort, production_prediction)
    exp_stage = _stage_metrics(cohort, exp141_prediction)

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
    return {"production": result, "exp141": payload}
