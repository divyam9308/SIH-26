"""Exp135: signed log-space Cost residual correction with variance recovery.

This challenger deliberately leaves the production Exp105/Exp113 module untouched until
canonical evaluation proves the replacement Cost correction. It uses the same Exp61 Cost
anchor, Exp105 factor features, forward OOF evidence, project-balanced weights, and Exp113
Delay path as production; only the Cost residual target/reconstruction changes to signed-log
space.
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
from backend.app.ml.monthly_training import _json_safe, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp35_baseline import (
    CALIBRATION_GATE_FEATURE,
    _aft_routing_limit,
    _select_aft_calibration_projects,
)
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors

EXPERIMENT_ID = "exp_135"
EXPERIMENT_NAME = "Log-Space Cost Regression with Variance Recovery"
COST_MODEL_ID = "exp61_plus_exp135_signed_log_cost_v1"


def signed_log(y: np.ndarray) -> np.ndarray:
    """Sign-preserving log1p transform for potentially negative Cost overrun percentages."""
    y_arr = np.asarray(y, dtype=float)
    return np.sign(y_arr) * np.log1p(np.abs(y_arr))


def inv_signed_log(z: np.ndarray) -> np.ndarray:
    """Inverse of :func:`signed_log`."""
    z_arr = np.asarray(z, dtype=float)
    return np.sign(z_arr) * np.expm1(np.abs(z_arr))


def _fit_log_residual_layer(
    oof: pd.DataFrame,
    score: pd.DataFrame,
    *,
    features: list[str],
    seed: int,
    min_fit_rows: int = 80,
    meta_estimators: int = 100,
    final_estimators: int = 140,
):
    """Fit the Cost residual booster in signed-log space using forward meta-OOF scale selection."""
    work = oof.copy()
    actual = pd.to_numeric(work["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(work["production_prediction"], errors="coerce").to_numpy(float)
    work["residual"] = signed_log(actual) - signed_log(anchor)

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
        _, _, x_fit, x_val = prod._numeric_design(fitting, validation, features)
        model = LGBMRegressor(
            n_estimators=meta_estimators,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=8,
            min_child_samples=60,
            reg_alpha=4,
            reg_lambda=20,
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

    if not meta_predictions:
        raise ValueError("Exp135 has no forward meta-OOF predictions")

    best = (float("inf"), 0.0)
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        fold_mae: list[float] = []
        fold_weight: list[float] = []
        for validation, correction in meta_predictions:
            anchor_raw = pd.to_numeric(
                validation["production_prediction"], errors="coerce"
            ).to_numpy(float)
            prediction = inv_signed_log(signed_log(anchor_raw) + scale * correction)
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
    cols, medians, x_fit, x_score = prod._numeric_design(work, score, features)
    model = LGBMRegressor(
        n_estimators=final_estimators,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=8,
        min_child_samples=60,
        reg_alpha=4,
        reg_lambda=20,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    residual = pd.to_numeric(work["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    weight = pd.to_numeric(work["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    model.fit(x_fit, residual, sample_weight=weight)
    cap = max(prod._weighted_quantile(np.abs(residual), weight, 0.90), 1e-9)
    score_correction = selected_scale * np.clip(
        np.asarray(model.predict(x_score), dtype=float), -cap, cap
    )
    score_anchor = pd.to_numeric(
        score["production_prediction"], errors="coerce"
    ).to_numpy(float)
    prediction = inv_signed_log(signed_log(score_anchor) + score_correction)
    return model, cols, medians, cap, selected_scale, prediction, years[1:]


def _exp135_training(
    train: pd.DataFrame,
    score: pd.DataFrame,
    cost_model,
    production_cost: np.ndarray,
):
    """Mirror Exp105 factor generation, replacing only residual target/reconstruction."""
    oof = prod._current_cost_oof(train, cost_model)
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    parts = []
    for year in sorted(int(v) for v in year_col.dropna().unique())[1:]:
        fitting = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(fitting) < 80 or validation.empty:
            continue
        scaler, factor, cols, medians = prod._fit_factor_transform(
            fitting, seed=10500 + year
        )
        z = prod._factor_transform(
            validation,
            scaler=scaler,
            factor=factor,
            cols=cols,
            medians=medians,
        )
        validation["exp105_factor_1"] = z[:, 0]
        validation["exp105_factor_2"] = z[:, 1]
        parts.append(validation)
    if not parts:
        raise ValueError("Exp135 has no forward factor folds")
    meta = pd.concat(parts, ignore_index=True)

    final_scaler, final_factor, factor_cols, factor_medians = prod._fit_factor_transform(
        oof, seed=10501
    )
    score_work = score.copy()
    score_work["production_prediction"] = np.asarray(production_cost, dtype=float)
    z = prod._factor_transform(
        score_work,
        scaler=final_scaler,
        factor=final_factor,
        cols=factor_cols,
        medians=factor_medians,
    )
    score_work["exp105_factor_1"] = z[:, 0]
    score_work["exp105_factor_2"] = z[:, 1]

    booster, booster_cols, booster_medians, cap, scale, prediction, meta_years = (
        _fit_log_residual_layer(
            meta,
            score_work,
            features=prod.EXP105_RESIDUAL_FEATURES,
            seed=13502,
        )
    )
    details = {
        "selected_scale": scale,
        "features": booster_cols,
        "medians": booster_medians,
        "cap_log_space": cap,
        "meta_oof_years": meta_years,
        "factor_inputs": prod.EXP105_FACTOR_INPUTS,
        "factor_fit_for_meta": "strictly earlier OOF years",
        "residual_space": "signed_log",
        "scale_selection_metric": "forward_meta_oof_mae_original_space",
    }
    return (
        final_scaler,
        final_factor,
        factor_cols,
        factor_medians,
        booster,
        booster_cols,
        booster_medians,
        cap,
        scale,
        prediction,
        details,
    )


class Exp135CostProductionModel:
    """Exp61 Cost anchor plus an Exp135 signed-log residual correction."""

    def __init__(
        self,
        *,
        base_model,
        factor_scaler,
        factor_model,
        factor_features,
        factor_medians,
        booster,
        booster_features,
        booster_medians,
        correction_cap,
        correction_scale,
        input_features,
    ):
        self.base_model = base_model
        self.factor_scaler = factor_scaler
        self.factor_model = factor_model
        self.factor_features = list(factor_features)
        self.factor_medians = dict(factor_medians)
        self.booster = booster
        self.booster_features = list(booster_features)
        self.booster_medians = dict(booster_medians)
        self.correction_cap = float(correction_cap)
        self.correction_scale = float(correction_scale)
        self.features = list(input_features)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        anchor = np.asarray(self.base_model.predict(frame), dtype=float)
        work = frame.copy()
        work["production_prediction"] = anchor
        z = prod._factor_transform(
            work,
            scaler=self.factor_scaler,
            factor=self.factor_model,
            cols=self.factor_features,
            medians=self.factor_medians,
        )
        work["exp105_factor_1"] = z[:, 0]
        work["exp105_factor_2"] = z[:, 1]
        x = prod._design_from_frozen(
            work, self.booster_features, self.booster_medians
        )
        corr_log = self.correction_scale * np.clip(
            np.asarray(self.booster.predict(x), dtype=float),
            -self.correction_cap,
            self.correction_cap,
        )
        return inv_signed_log(signed_log(anchor) + corr_log)


def train_window_with_exp135(
    training_start: int,
    training_end: int,
    test_end: int,
    *,
    data: pd.DataFrame,
    identity: pd.DataFrame | None = None,
    artifact_root: Path,
) -> dict:
    """Train current production stack in an isolated root, then replace Cost with Exp135."""
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
    if not isinstance(current_cost_model, prod.Exp105CostProductionModel):
        raise TypeError("Exp135 expects the current Exp105 Cost wrapper as its comparison model")
    anchor_model = current_cost_model.base_model

    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(
        prepared, training_start, training_end, test_end
    )
    prior_train, prior_test, _ = _build_temporal_delay_priors(train, test)
    cohort = _production_cost_evaluation_rows(prior_test).copy()
    calibration_ids = _select_aft_calibration_projects(
        cohort,
        limit=_aft_routing_limit(training_start, training_end, test_end),
    )
    cohort[CALIBRATION_GATE_FEATURE] = cohort["canonical_project_id"].astype(
        "string"
    ).isin(calibration_ids)
    cohort = assign_project_balanced_weights(cohort)

    anchor_prediction = np.asarray(anchor_model.predict(cohort), dtype=float)
    production_prediction = np.asarray(current_cost_model.predict(cohort), dtype=float)
    (
        factor_scaler,
        factor_model,
        factor_features,
        factor_medians,
        booster,
        booster_features,
        booster_medians,
        cap,
        scale,
        exp135_prediction,
        details,
    ) = _exp135_training(prior_train, cohort, anchor_model, anchor_prediction)

    metadata = dict(result.get("metadata") or {})
    contract = prod.target_feature_contract(metadata)
    cost_input_features = list(
        dict.fromkeys(list(contract["cost"]) + prod.EXP105_FACTOR_INPUTS)
    )
    model = Exp135CostProductionModel(
        base_model=anchor_model,
        factor_scaler=factor_scaler,
        factor_model=factor_model,
        factor_features=factor_features,
        factor_medians=factor_medians,
        booster=booster,
        booster_features=booster_features,
        booster_medians=booster_medians,
        correction_cap=cap,
        correction_scale=scale,
        input_features=cost_input_features,
    )
    contract_prediction = model.predict(cohort.reindex(columns=cost_input_features))
    if not np.allclose(contract_prediction, exp135_prediction, rtol=0.0, atol=1e-9):
        raise AssertionError("Exp135 wrapper diverged from experiment score path")

    current_metrics = prod._metric(
        cohort, "actual_cost_overrun_percentage", production_prediction
    )
    exp135_metrics = prod._metric(
        cohort, "actual_cost_overrun_percentage", exp135_prediction
    )
    anchor_metrics = prod._metric(
        cohort, "actual_cost_overrun_percentage", anchor_prediction
    )

    joblib.dump(model, target / "cost_model.pkl")
    metadata["exp135_cost_challenger"] = {
        **_json_safe(details),
        "holdout_used_for_fit_or_selection": False,
        "comparison_production_model": prod.PRODUCTION_COST_BASELINE,
        "challenger_model": COST_MODEL_ID,
        "anchor_metrics": anchor_metrics,
        "production_metrics": current_metrics,
        "challenger_metrics": exp135_metrics,
    }
    result["metadata"] = metadata
    result["exp135"] = {
        "experiment_id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "training_start": training_start,
        "training_end": training_end,
        "test_start": training_end + 1,
        "test_end": test_end,
        "production_cost_metrics": current_metrics,
        "experiment_cost_metrics": exp135_metrics,
        "mae_delta": float(exp135_metrics["MAE"]) - float(current_metrics["MAE"]),
        "rmse_delta": float(exp135_metrics["RMSE"]) - float(current_metrics["RMSE"]),
        "r2_delta": float(exp135_metrics["R2"]) - float(current_metrics["R2"]),
        "selected_scale": scale,
        "holdout_used_for_selection": False,
        "scientific_verdict": (
            "PROMOTION CANDIDATE"
            if float(exp135_metrics["MAE"]) <= float(current_metrics["MAE"])
            and float(exp135_metrics["RMSE"]) <= float(current_metrics["RMSE"])
            and float(exp135_metrics["R2"]) > float(current_metrics["R2"])
            else "DO NOT PROMOTE"
        ),
    }
    safe = _json_safe(result)
    (target / "metadata.json").write_text(
        json.dumps(safe["metadata"], indent=2, allow_nan=False)
    )
    (target / "evaluation_results.json").write_text(
        json.dumps(safe, indent=2, allow_nan=False)
    )
    return safe
