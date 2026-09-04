"""Production promotion of Exp105 Cost and Exp113 Delay.

Training starts from the current Exp61 + U1 production stack, then applies only:
- Exp105 dynamic multivariate execution-factor residual correction to Cost.
- Exp113 quantile-AFT uncertainty residual correction to Delay.

Risk is retained unchanged. Both promotion layers are selected/fitted exclusively
from forward training-only OOF evidence and preserve production as the anchor.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _aft_remaining_prediction,
    _corrections,
    _delay_from_remaining,
    _fit_aft_family_models,
    _remaining_frame,
)
from backend.app.ml.experiments.nextgen_common import (
    _family,
    _prepare,
    normalize_taxonomy,
    shrunk_calibration,
)
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import (
    MODEL_ROOT,
    _fit_pipeline,
    _json_safe,
    _regression_metrics,
    _regressors,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _prediction_rows,
    _production_cost_evaluation_rows,
    target_feature_contract,
)
from backend.app.ml.production_exp35_baseline import (
    CALIBRATION_GATE_FEATURE,
    _aft_routing_limit,
    _select_aft_calibration_projects,
)
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_u1_delay_baseline import (
    _fit_u1_booster,
    _u1_prior_enrich,
    train_window_with_promoted_cost_and_delay as train_u1_production,
)
from backend.app.ml.provenance import (
    artifact_fingerprints,
    feature_schema_fingerprint,
    file_sha256,
)

PROMOTED_COST_EXPERIMENT_ID = "exp_105"
PROMOTED_DELAY_EXPERIMENT_ID = "exp_113"
PRODUCTION_COST_BASELINE = "exp61_plus_exp105_dynamic_factor_cost_v1"
PRODUCTION_DELAY_BASELINE = "exp61_u1_plus_exp113_quantile_aft_delay_v1"

EXP105_FACTOR_INPUTS = [
    "cost_escalation_percentage",
    "expenditure_ratio",
    "progress_deviation",
    "schedule_slippage_days",
    "duration_ratio",
    "physical_progress",
]
EXP105_RESIDUAL_FEATURES = [
    "production_prediction",
    "exp105_factor_1",
    "exp105_factor_2",
    "cost_escalation_percentage",
    "duration_ratio",
]

EXP113_QUANTILES = (0.25, 0.5, 0.75)
EXP113_QUANTILE_BASE_FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "expenditure_ratio",
    "cost_escalation_percentage",
    "progress_deviation",
    "approved_cost_cr",
    "planned_duration_days",
    "elapsed_duration_days",
    "exp58_delay_hier_prior",
    "exp58_group_support",
]
EXP113_RESIDUAL_FEATURES = [
    "production_prediction",
    "u1_correction",
    "exp113_q50_delay",
    "exp113_interval_width",
    "exp113_upper_asymmetry",
    "exp113_lower_asymmetry",
    "duration_ratio",
    "schedule_slippage_days",
    "expenditure_ratio",
    "cost_escalation_percentage",
    "exp58_group_support",
]

_FINGERPRINTED_ARTIFACTS = [
    "cost_model.pkl",
    "delay_model.pkl",
    "risk_model.pkl",
    "feature_quality_report.json",
    "shap_importance.json",
    "prediction_validation.csv",
]


def _metric(frame: pd.DataFrame, actual: str, prediction: np.ndarray) -> dict:
    return _regression_metrics(
        frame[actual], prediction, frame["sample_weight"], frame["canonical_project_id"]
    )


def _gain(base: float, candidate: float) -> float:
    return (float(base) - float(candidate)) / float(base) * 100.0 if float(base) else 0.0


def _weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[mask], weights[mask]
    if not len(values):
        return 0.0
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total = float(weights.sum())
    if total <= 0:
        return float(np.quantile(values, q))
    return float(values[np.searchsorted(np.cumsum(weights), q * total, side="left")])


def _weighted_mae(actual, prediction, weights) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (
        np.isfinite(actual)
        & np.isfinite(prediction)
        & np.isfinite(weights)
        & (weights >= 0)
    )
    if not mask.any():
        return float("inf")
    w = weights[mask]
    err = np.abs(actual[mask] - prediction[mask])
    return float(np.average(err, weights=w)) if float(w.sum()) > 0 else float(np.mean(err))


def _numeric_design(train: pd.DataFrame, score: pd.DataFrame, features: list[str]):
    cols = [c for c in features if c in train.columns and c in score.columns]
    medians: dict[str, float] = {}
    left: dict[str, pd.Series] = {}
    right: dict[str, pd.Series] = {}
    for col in cols:
        a = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(score[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(a.median()) if a.notna().any() else 0.0
        medians[col] = median
        left[col] = a.fillna(median)
        right[col] = b.fillna(median)
    return cols, medians, pd.DataFrame(left, index=train.index), pd.DataFrame(right, index=score.index)


def _design_from_frozen(frame: pd.DataFrame, features: list[str], medians: dict[str, float]) -> pd.DataFrame:
    columns = {}
    for col in features:
        values = pd.to_numeric(
            frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        columns[col] = values.fillna(float(medians.get(col, 0.0)))
    return pd.DataFrame(columns, index=frame.index)


def _forward_folds(frame: pd.DataFrame, max_folds: int):
    completion_year = pd.to_numeric(frame["completion_year"], errors="coerce")
    years = sorted(int(x) for x in completion_year.dropna().unique())
    out = []
    for year in reversed(years[1:]):
        fitting = frame.loc[completion_year < year].copy()
        validation = frame.loc[completion_year == year].copy()
        if fitting["canonical_project_id"].nunique() >= 10 and validation["canonical_project_id"].nunique() >= 3:
            out.append((fitting, validation, year))
        if len(out) >= max_folds:
            break
    return list(reversed(out))


def _raw_cost_oof(frame: pd.DataFrame, features: list[str], family: str, max_folds: int = 3) -> pd.DataFrame:
    out = []
    for fitting, validation, year in _forward_folds(frame, max_folds):
        model = _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[family], fitting, features, "actual_cost_overrun_percentage"
        )
        raw = np.asarray(model.predict(validation.reindex(columns=features)), dtype=float)
        part = validation.copy()
        part["prediction"] = raw
        part["residual"] = pd.to_numeric(part["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float) - raw
        part["oof_year"] = int(year)
        out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _current_cost_oof(train: pd.DataFrame, production_model) -> pd.DataFrame:
    features = list(production_model.features)
    family = _family(production_model)
    out = []
    for fitting, validation, year in _forward_folds(train, 6):
        inner = _raw_cost_oof(fitting, features, family, 3)
        if inner.empty:
            continue
        calibration = shrunk_calibration(inner, 40.0)
        model = _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[family], fitting, features, "actual_cost_overrun_percentage"
        )
        raw = np.asarray(model.predict(validation.reindex(columns=features)), dtype=float)
        prediction = raw + _corrections(validation, raw, calibration)
        part = validation.copy()
        part["production_prediction"] = prediction
        part["residual"] = pd.to_numeric(part["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float) - prediction
        part["oof_year"] = int(year)
        out.append(part)
    if len(out) < 2:
        raise ValueError("Exp105 production promotion requires at least two forward Cost OOF folds")
    return pd.concat(out, ignore_index=True)


def _base_delay_oof(train: pd.DataFrame, u1_model) -> pd.DataFrame:
    if not hasattr(u1_model, "base_model"):
        raise TypeError("Exp113 production promotion requires the U1 Delay wrapper")
    base = u1_model.base_model
    features = list(base.model_features)
    train_delay = _remaining_frame(train)
    out = []
    for fitting, validation, year in _forward_folds(train_delay, 8):
        models = _fit_aft_family_models(fitting, features)
        remaining = _aft_remaining_prediction(models, base.weights, validation, features)
        raw = _delay_from_remaining(validation, remaining)
        prediction = np.maximum(0.0, raw + _corrections(validation, raw, base.calibration))
        part = validation.copy()
        part["base_prediction"] = prediction
        part["production_prediction"] = prediction
        part["residual"] = pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float) - prediction
        part["oof_year"] = int(year)
        out.append(part)
    if len(out) < 4:
        raise ValueError("Exp113 production promotion requires at least four base Delay OOF folds")
    return pd.concat(out, ignore_index=True)


def _current_delay_oof(train: pd.DataFrame, u1_model) -> pd.DataFrame:
    base = _base_delay_oof(train, u1_model)
    years_col = pd.to_numeric(base["oof_year"], errors="coerce")
    years = sorted(int(x) for x in years_col.dropna().unique())
    out = []
    for year in years[1:]:
        fitting = base.loc[years_col < year].copy()
        validation = base.loc[years_col == year].copy()
        if len(fitting) < 100 or validation.empty:
            continue
        _, _, _, _, correction = _fit_u1_booster(fitting, validation)
        anchor = pd.to_numeric(validation["base_prediction"], errors="coerce").to_numpy(float)
        prediction = np.maximum(0.0, anchor + correction)
        part = validation.copy()
        part["production_prediction"] = prediction
        part["u1_correction"] = prediction - anchor
        part["residual"] = pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float) - prediction
        part["oof_year"] = int(year)
        out.append(part)
    if len(out) < 3:
        raise ValueError("Exp113 production promotion requires at least three current Delay OOF folds")
    return pd.concat(out, ignore_index=True)


def _fit_residual_layer(
    oof: pd.DataFrame,
    score: pd.DataFrame,
    *,
    features: list[str],
    seed: int,
    actual_col: str,
    min_fit_rows: int,
    meta_estimators: int,
    final_estimators: int,
    nonnegative_output: bool,
):
    years = sorted(int(x) for x in pd.to_numeric(oof["oof_year"], errors="coerce").dropna().unique())
    meta_predictions = []
    for year in years[1:]:
        year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
        fitting = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(fitting) < min_fit_rows or validation.empty:
            continue
        _, _, x_fit, x_val = _numeric_design(fitting, validation, features)
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
        cap = max(_weighted_quantile(np.abs(residual), weight, 0.90), 1e-9)
        correction = np.clip(np.asarray(model.predict(x_val), dtype=float), -cap, cap)
        meta_predictions.append((validation, correction))
    if not meta_predictions:
        raise ValueError("Promotion residual layer has no forward meta-OOF predictions")

    best = (float("inf"), 0.0)
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        fold_mae = []
        fold_weight = []
        for validation, correction in meta_predictions:
            anchor = pd.to_numeric(validation["production_prediction"], errors="coerce").to_numpy(float)
            prediction = anchor + scale * correction
            if nonnegative_output:
                prediction = np.maximum(0.0, prediction)
            actual = pd.to_numeric(validation[actual_col], errors="coerce").to_numpy(float)
            weight = pd.to_numeric(validation["sample_weight"], errors="coerce").to_numpy(float)
            fold_mae.append(_weighted_mae(actual, prediction, weight))
            fold_weight.append(max(float(np.nansum(weight)), 1e-9))
        candidate = (float(np.average(fold_mae, weights=fold_weight)), scale)
        if candidate < best:
            best = candidate

    selected_scale = float(best[1])
    cols, medians, x_fit, x_score = _numeric_design(oof, score, features)
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
    residual = pd.to_numeric(oof["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    model.fit(x_fit, residual, sample_weight=weight)
    cap = max(_weighted_quantile(np.abs(residual), weight, 0.90), 1e-9)
    correction = selected_scale * np.clip(np.asarray(model.predict(x_score), dtype=float), -cap, cap)
    return model, cols, medians, cap, selected_scale, correction, years[1:]


def _fit_factor_transform(train: pd.DataFrame, *, seed: int):
    cols, medians, x_train, _ = _numeric_design(train, train, EXP105_FACTOR_INPUTS)
    scaler = StandardScaler()
    z_train = scaler.fit_transform(x_train)
    factor = FactorAnalysis(n_components=2, random_state=seed, max_iter=500)
    factor.fit(z_train)
    return scaler, factor, cols, medians


def _factor_transform(frame: pd.DataFrame, *, scaler, factor, cols: list[str], medians: dict[str, float]) -> np.ndarray:
    x = _design_from_frozen(frame, cols, medians)
    return factor.transform(scaler.transform(x))


def _exp105_training(train: pd.DataFrame, score: pd.DataFrame, cost_model, production_cost: np.ndarray):
    oof = _current_cost_oof(train, cost_model)
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    parts = []
    for year in sorted(int(v) for v in year_col.dropna().unique())[1:]:
        fitting = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(fitting) < 80 or validation.empty:
            continue
        scaler, factor, cols, medians = _fit_factor_transform(fitting, seed=10500 + year)
        z = _factor_transform(validation, scaler=scaler, factor=factor, cols=cols, medians=medians)
        validation["exp105_factor_1"] = z[:, 0]
        validation["exp105_factor_2"] = z[:, 1]
        parts.append(validation)
    if not parts:
        raise ValueError("Exp105 production promotion has no forward factor folds")
    meta = pd.concat(parts, ignore_index=True)

    final_scaler, final_factor, factor_cols, factor_medians = _fit_factor_transform(oof, seed=10501)
    score_work = score.copy()
    score_work["production_prediction"] = np.asarray(production_cost, dtype=float)
    z = _factor_transform(score_work, scaler=final_scaler, factor=final_factor, cols=factor_cols, medians=factor_medians)
    score_work["exp105_factor_1"] = z[:, 0]
    score_work["exp105_factor_2"] = z[:, 1]

    booster, booster_cols, booster_medians, cap, scale, correction, meta_years = _fit_residual_layer(
        meta,
        score_work,
        features=EXP105_RESIDUAL_FEATURES,
        seed=10502,
        actual_col="actual_cost_overrun_percentage",
        min_fit_rows=80,
        meta_estimators=100,
        final_estimators=140,
        nonnegative_output=False,
    )
    details = {
        "selected_scale": scale,
        "features": booster_cols,
        "medians": booster_medians,
        "cap": cap,
        "meta_oof_years": meta_years,
        "factor_inputs": EXP105_FACTOR_INPUTS,
        "factor_fit_for_meta": "strictly earlier OOF years",
    }
    return final_scaler, final_factor, factor_cols, factor_medians, booster, booster_cols, booster_medians, cap, scale, correction, details


def _remaining_target(frame: pd.DataFrame) -> np.ndarray:
    completion = pd.to_datetime(frame["completion_date"], errors="coerce")
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    return (completion - snapshot).dt.days.clip(lower=1).astype(float).to_numpy(float)


def _fit_quantile_models(train: pd.DataFrame, *, seed: int):
    cols, medians, x_train, _ = _numeric_design(train, train, EXP113_QUANTILE_BASE_FEATURES)
    target = np.log1p(_remaining_target(train))
    weight = pd.to_numeric(train["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    models = {}
    for i, quantile in enumerate(EXP113_QUANTILES):
        model = LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=160,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=10,
            min_child_samples=70,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=5,
            reg_lambda=25,
            random_state=seed + i,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(x_train, target, sample_weight=weight)
        models[quantile] = model
    return models, cols, medians


def _score_quantiles(score: pd.DataFrame, *, models: dict, cols: list[str], medians: dict[str, float]) -> dict[float, np.ndarray]:
    x = _design_from_frozen(score, cols, medians)
    out = {}
    for quantile, model in models.items():
        log_remaining = np.asarray(model.predict(x), dtype=float)
        out[float(quantile)] = np.maximum(1.0, np.expm1(np.clip(log_remaining, -10, 10)))
    return out


def _attach_exp113_quantiles(train: pd.DataFrame, score: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    models, cols, medians = _fit_quantile_models(train, seed=seed)
    quantile = _score_quantiles(score, models=models, cols=cols, medians=medians)
    result = score.copy()
    result["exp113_q50_delay"] = _delay_from_remaining(result, quantile[0.5])
    result["exp113_interval_width"] = quantile[0.75] - quantile[0.25]
    result["exp113_upper_asymmetry"] = quantile[0.75] - quantile[0.5]
    result["exp113_lower_asymmetry"] = quantile[0.5] - quantile[0.25]
    return result


def _exp113_training(train: pd.DataFrame, score: pd.DataFrame, delay_model, production_delay: np.ndarray, training_end: int):
    oof = _current_delay_oof(train, delay_model)
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(x) for x in year_col.dropna().unique())
    parts = []
    for year in years[1:]:
        fitting = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(fitting) < 120 or validation.empty:
            continue
        parts.append(_attach_exp113_quantiles(fitting, validation, seed=11300 + year))
    if not parts:
        raise ValueError("Exp113 production promotion has no forward quantile-AFT evidence")
    meta = pd.concat(parts, ignore_index=True)

    score_work = score.copy()
    score_work["production_prediction"] = np.asarray(production_delay, dtype=float)
    base_prediction = np.maximum(0.0, np.asarray(delay_model.base_model.predict(score_work), dtype=float))
    score_work["u1_correction"] = np.asarray(production_delay, dtype=float) - base_prediction

    quantile_models, quantile_cols, quantile_medians = _fit_quantile_models(oof, seed=11400 + training_end)
    quantile = _score_quantiles(score_work, models=quantile_models, cols=quantile_cols, medians=quantile_medians)
    score_work["exp113_q50_delay"] = _delay_from_remaining(score_work, quantile[0.5])
    score_work["exp113_interval_width"] = quantile[0.75] - quantile[0.25]
    score_work["exp113_upper_asymmetry"] = quantile[0.75] - quantile[0.5]
    score_work["exp113_lower_asymmetry"] = quantile[0.5] - quantile[0.25]

    booster, booster_cols, booster_medians, cap, scale, correction, meta_years = _fit_residual_layer(
        meta,
        score_work,
        features=EXP113_RESIDUAL_FEATURES,
        seed=11301,
        actual_col="actual_delay_days",
        min_fit_rows=100,
        meta_estimators=120,
        final_estimators=160,
        nonnegative_output=True,
    )
    details = {
        "selected_scale": scale,
        "features": booster_cols,
        "medians": booster_medians,
        "cap": cap,
        "meta_oof_years": meta_years,
        "quantiles": list(EXP113_QUANTILES),
        "quantile_meta_predictions_are_forward_oof": True,
    }
    return quantile_models, quantile_cols, quantile_medians, booster, booster_cols, booster_medians, cap, scale, correction, details


class Exp105CostProductionModel:
    """Current Cost anchor plus the frozen Exp105 factor-residual correction."""

    def __init__(self, *, base_model, factor_scaler, factor_model, factor_features, factor_medians, booster, booster_features, booster_medians, correction_cap, correction_scale, input_features):
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
        z = _factor_transform(work, scaler=self.factor_scaler, factor=self.factor_model, cols=self.factor_features, medians=self.factor_medians)
        work["exp105_factor_1"] = z[:, 0]
        work["exp105_factor_2"] = z[:, 1]
        x = _design_from_frozen(work, self.booster_features, self.booster_medians)
        correction = self.correction_scale * np.clip(
            np.asarray(self.booster.predict(x), dtype=float), -self.correction_cap, self.correction_cap
        )
        return anchor + correction


class Exp113DelayProductionModel:
    """Current U1 Delay anchor plus the frozen Exp113 quantile-AFT correction."""

    def __init__(self, *, base_model, quantile_models, quantile_features, quantile_medians, booster, booster_features, booster_medians, correction_cap, correction_scale, input_features):
        self.base_model = base_model
        self.quantile_models = dict(quantile_models)
        self.quantile_features = list(quantile_features)
        self.quantile_medians = dict(quantile_medians)
        self.booster = booster
        self.booster_features = list(booster_features)
        self.booster_medians = dict(booster_medians)
        self.correction_cap = float(correction_cap)
        self.correction_scale = float(correction_scale)
        self.features = list(input_features)
        self.model_features = list(getattr(base_model, "model_features", []))

    def _enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        prior_state = getattr(self.base_model, "booster_prior_state", None)
        return _u1_prior_enrich(frame.copy(), prior_state) if prior_state else frame.copy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        anchor = np.maximum(0.0, np.asarray(self.base_model.predict(frame), dtype=float))
        work = self._enrich(frame)
        work["production_prediction"] = anchor
        base_anchor = np.maximum(0.0, np.asarray(self.base_model.base_model.predict(frame), dtype=float))
        work["u1_correction"] = anchor - base_anchor
        quantile = _score_quantiles(work, models=self.quantile_models, cols=self.quantile_features, medians=self.quantile_medians)
        work["exp113_q50_delay"] = _delay_from_remaining(work, quantile[0.5])
        work["exp113_interval_width"] = quantile[0.75] - quantile[0.25]
        work["exp113_upper_asymmetry"] = quantile[0.75] - quantile[0.5]
        work["exp113_lower_asymmetry"] = quantile[0.5] - quantile[0.25]
        x = _design_from_frozen(work, self.booster_features, self.booster_medians)
        correction = self.correction_scale * np.clip(
            np.asarray(self.booster.predict(x), dtype=float), -self.correction_cap, self.correction_cap
        )
        return np.maximum(0.0, anchor + correction)


def train_window_with_promoted_cost_and_delay(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
    verify_frozen_reference: bool = True,
) -> dict:
    result = train_u1_production(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=artifact_root,
        verify_frozen_reference=verify_frozen_reference,
    )
    if data is None:
        raise ValueError("Exp105 + Exp113 production promotion requires the supervised frame")

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"
    metadata = dict(result.get("metadata") or {})
    contract = target_feature_contract(metadata)

    base_cost_model = joblib.load(target / "cost_model.pkl")
    base_delay_model = joblib.load(target / "delay_model.pkl")
    risk_model = joblib.load(target / "risk_model.pkl")
    risk_hash_before = file_sha256(target / "risk_model.pkl")

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

    old_cost = np.asarray(base_cost_model.predict(cohort), dtype=float)
    old_delay = np.maximum(0.0, np.asarray(base_delay_model.predict(cohort), dtype=float))

    factor_scaler, factor_model, factor_features, factor_medians, cost_booster, cost_booster_features, cost_booster_medians, cost_cap, cost_scale, cost_correction, exp105_details = _exp105_training(
        prior_train, cohort, base_cost_model, old_cost
    )
    cost_input_features = list(dict.fromkeys(list(contract["cost"]) + EXP105_FACTOR_INPUTS))
    promoted_cost_model = Exp105CostProductionModel(
        base_model=base_cost_model,
        factor_scaler=factor_scaler,
        factor_model=factor_model,
        factor_features=factor_features,
        factor_medians=factor_medians,
        booster=cost_booster,
        booster_features=cost_booster_features,
        booster_medians=cost_booster_medians,
        correction_cap=cost_cap,
        correction_scale=cost_scale,
        input_features=cost_input_features,
    )
    promoted_cost = old_cost + cost_correction

    quantile_models, quantile_features, quantile_medians, delay_booster, delay_booster_features, delay_booster_medians, delay_cap, delay_scale, delay_correction, exp113_details = _exp113_training(
        prior_train, cohort, base_delay_model, old_delay, training_end
    )
    delay_extra_inputs = [
        "snapshot_date",
        "planned_completion_date",
        "duration_ratio",
        "schedule_slippage_days",
        "expenditure_ratio",
        "cost_escalation_percentage",
        "progress_deviation",
        "approved_cost_cr",
        "planned_duration_days",
        "elapsed_duration_days",
        "sector",
        "implementing_agency",
    ]
    delay_input_features = list(dict.fromkeys(list(contract["delay"]) + delay_extra_inputs))
    promoted_delay_model = Exp113DelayProductionModel(
        base_model=base_delay_model,
        quantile_models=quantile_models,
        quantile_features=quantile_features,
        quantile_medians=quantile_medians,
        booster=delay_booster,
        booster_features=delay_booster_features,
        booster_medians=delay_booster_medians,
        correction_cap=delay_cap,
        correction_scale=delay_scale,
        input_features=delay_input_features,
    )
    promoted_delay = np.maximum(0.0, old_delay + delay_correction)

    cost_contract_prediction = promoted_cost_model.predict(cohort.reindex(columns=cost_input_features))
    delay_contract_prediction = promoted_delay_model.predict(cohort.reindex(columns=delay_input_features))
    if not np.allclose(cost_contract_prediction, promoted_cost, rtol=0.0, atol=1e-9):
        raise AssertionError("Exp105 production wrapper diverged from experiment score path")
    if not np.allclose(delay_contract_prediction, promoted_delay, rtol=0.0, atol=1e-9):
        raise AssertionError("Exp113 production wrapper diverged from experiment score path")

    old_cost_metrics = _metric(cohort, "actual_cost_overrun_percentage", old_cost)
    cost_metrics = _metric(cohort, "actual_cost_overrun_percentage", promoted_cost)
    old_delay_metrics = _metric(cohort, "actual_delay_days", old_delay)
    delay_metrics = _metric(cohort, "actual_delay_days", promoted_delay)

    if (training_start, training_end, test_end) == (2001, 2021, 2025):
        if float(cost_metrics["MAE"]) >= float(old_cost_metrics["MAE"]):
            raise RuntimeError("Exp105 Cost failed to improve the verified 2001-2021 production window")
        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):
            raise RuntimeError("Exp113 Delay failed to improve the verified 2001-2021 production window")

    prior_test = prior_test.copy()
    prior_test[CALIBRATION_GATE_FEATURE] = prior_test["canonical_project_id"].astype("string").isin(calibration_ids)
    joblib.dump(promoted_cost_model, target / "cost_model.pkl")
    joblib.dump(promoted_delay_model, target / "delay_model.pkl")
    if file_sha256(target / "risk_model.pkl") != risk_hash_before:
        raise AssertionError("Exp105 + Exp113 promotion modified Risk artifact")

    risk_features = list(contract["risk"])
    computed_cost_metrics, validation_rows, cost_evaluation_contract = _prediction_rows(
        prior_test,
        cost_model=promoted_cost_model,
        cost_features=cost_input_features,
        delay_model=promoted_delay_model,
        delay_features=delay_input_features,
        risk_model=risk_model,
        risk_features=risk_features,
    )
    if abs(float(computed_cost_metrics["MAE"]) - float(cost_metrics["MAE"])) > 1e-9:
        raise AssertionError("Exp105 production Cost evaluation diverged after serialization")
    validation_rows.to_csv(target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")

    metadata["base_production_cost_baseline"] = metadata.get("production_cost_baseline")
    metadata["base_production_delay_baseline"] = metadata.get("production_delay_baseline")
    metadata["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    metadata["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    metadata["promoted_cost_from_experiment"] = PROMOTED_COST_EXPERIMENT_ID
    metadata["promoted_delay_from_experiment"] = PROMOTED_DELAY_EXPERIMENT_ID
    metadata["promotion_scope"] = "cost+delay"
    metadata["cost_policy"] = "Exp61 Cost plus Exp105 forward-fitted dynamic execution-factor residual correction"
    metadata["delay_policy"] = "Exp61 + U1 Delay plus Exp113 forward-OOF quantile-AFT uncertainty residual correction"
    metadata["risk_policy"] = "existing_production_retained"
    metadata["cost_features_used"] = cost_input_features
    metadata["delay_features_used"] = delay_input_features
    metadata["risk_features_used"] = risk_features
    metadata["feature_count_by_target"] = {
        "cost": len(cost_input_features), "delay": len(delay_input_features), "risk": len(risk_features)
    }
    metadata["exp105_cost_promotion"] = {
        **_json_safe(exp105_details), "holdout_used_for_fit_or_selection": False, "base_prediction_replaced": False
    }
    metadata["exp113_delay_promotion"] = {
        **_json_safe(exp113_details), "holdout_used_for_fit_or_selection": False, "base_prediction_replaced": False
    }
    metadata.setdefault("lifecycle_metrics", {})["cost"] = cost_metrics
    metadata.setdefault("lifecycle_metrics", {})["delay"] = delay_metrics
    metadata["cost_evaluation_contract"] = cost_evaluation_contract
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["cost"] = "exp61_cost_plus_exp105_dynamic_factor_residual"
    selected["delay"] = "exp61_u1_delay_plus_exp113_quantile_aft_residual"
    metadata["selected_algorithms"] = selected
    metadata["leakage_policy"] = (
        str(metadata.get("leakage_policy") or "")
        + " Exp105 Cost factor transforms and residual booster are fitted only from forward training OOF evidence. "
          "Exp113 quantile-AFT features and its residual booster are also generated/fitted strictly forward OOF. "
          "Future holdout outcomes are never used for fitting, feature transformation, residual-scale selection, or quantile training."
    ).strip()

    lifecycle = dict(result.get("lifecycle") or {})
    lifecycle.setdefault("metrics", {})["cost"] = cost_metrics
    lifecycle.setdefault("metrics", {})["delay"] = delay_metrics
    lifecycle["target_features"] = {
        "cost": cost_input_features, "delay": delay_input_features, "risk": risk_features
    }
    lifecycle["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    lifecycle["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    result["lifecycle"] = lifecycle
    result["promotion"] = {
        "scope": "cost+delay",
        "cost": {
            "experiment_id": PROMOTED_COST_EXPERIMENT_ID,
            "previous_cost_mae": old_cost_metrics["MAE"],
            "promoted_cost_mae": cost_metrics["MAE"],
            "cost_improvement_percentage": round(_gain(old_cost_metrics["MAE"], cost_metrics["MAE"]), 6),
        },
        "delay": {
            "experiment_id": PROMOTED_DELAY_EXPERIMENT_ID,
            "previous_delay_mae": old_delay_metrics["MAE"],
            "promoted_delay_mae": delay_metrics["MAE"],
            "delay_improvement_percentage": round(_gain(old_delay_metrics["MAE"], delay_metrics["MAE"]), 6),
        },
        "risk_retained": True,
    }

    provenance = dict(metadata.get("provenance") or {})
    provenance["feature_schema_fingerprint"] = feature_schema_fingerprint(
        list(dict.fromkeys(cost_input_features + delay_input_features + risk_features))
    )
    provenance["artifact_fingerprints"] = artifact_fingerprints(target, _FINGERPRINTED_ARTIFACTS)
    metadata["provenance"] = provenance
    result["metadata"] = metadata

    result = _json_safe(result)
    (target / "metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False))
    (target / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
