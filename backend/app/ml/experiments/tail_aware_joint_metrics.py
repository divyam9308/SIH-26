"""Tail-aware residual correction experiment for Cost and Delay.

This module is diagnostic/experimental only. It learns correction layers from
forward temporal OOF evidence inside the 2001-2021 training period and never
uses 2022-2025 outcomes to select a correction, scale, threshold, or feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


COST_FEATURES = [
    "production_prediction",
    "cost_escalation_percentage",
    "expenditure_ratio",
    "progress_deviation",
    "schedule_slippage_days",
    "duration_ratio",
    "physical_progress",
    "approved_cost_cr",
    "planned_duration_days",
    "elapsed_duration_days",
]

DELAY_FEATURES = [
    "production_prediction",
    "schedule_slippage_days",
    "duration_ratio",
    "expenditure_ratio",
    "cost_escalation_percentage",
    "progress_deviation",
    "approved_cost_cr",
    "planned_duration_days",
    "elapsed_duration_days",
    "physical_progress",
]


@dataclass
class TailAwareLayer:
    model: LGBMRegressor
    features: list[str]
    medians: dict[str, float]
    scale: float
    correction_cap: float
    p90: float
    p95: float
    target: str


def _numeric(frame: pd.DataFrame, features: list[str], medians: dict[str, float] | None = None):
    medians = {} if medians is None else dict(medians)
    values = {}
    for col in features:
        s = pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col not in medians:
            medians[col] = float(s.median()) if s.notna().any() else 0.0
        values[col] = s.fillna(medians[col])
    return pd.DataFrame(values, index=frame.index), medians


def _weighted_metrics(actual, prediction, weight) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    w = np.asarray(weight, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w >= 0)
    y, p, w = y[mask], p[mask], w[mask]
    if not len(y):
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    if float(w.sum()) <= 0:
        w = np.ones_like(y)
    mae = float(np.average(np.abs(y - p), weights=w))
    mse = float(np.average((y - p) ** 2, weights=w))
    mean = float(np.average(y, weights=w))
    denom = float(np.sum(w * (y - mean) ** 2))
    r2 = 1.0 - float(np.sum(w * (y - p) ** 2)) / denom if denom > 0 else float("nan")
    return {"MAE": mae, "RMSE": float(np.sqrt(mse)), "R2": r2}


def _tail_weights(actual: pd.Series, base_weight: pd.Series):
    y = pd.to_numeric(actual, errors="coerce")
    unique = y.dropna()
    if len(unique) < 20:
        raise ValueError("Tail-aware experiment requires at least 20 training observations")
    p90 = float(unique.quantile(0.90))
    p95 = float(unique.quantile(0.95))
    mult = np.where(y > p95, 3.0, np.where(y > p90, 1.75, 1.0))
    w = pd.to_numeric(base_weight, errors="coerce").fillna(0.0).to_numpy(float) * mult
    return w, p90, p95


def _objective(base: dict[str, float], candidate: dict[str, float]) -> float:
    # Optimize all three requested metrics. R2 regression is explicitly penalized.
    mae_ratio = candidate["MAE"] / max(base["MAE"], 1e-9)
    rmse_ratio = candidate["RMSE"] / max(base["RMSE"], 1e-9)
    r2_penalty = max(0.0, base["R2"] - candidate["R2"])
    r2_reward = max(0.0, candidate["R2"] - base["R2"])
    return 0.45 * mae_ratio + 0.45 * rmse_ratio + 1.5 * r2_penalty - 0.10 * r2_reward


def fit_tail_aware_layer(
    oof: pd.DataFrame,
    *,
    actual_col: str,
    features: list[str],
    target: str,
    seed: int,
    nonnegative: bool,
) -> TailAwareLayer:
    required = {"production_prediction", "oof_year", actual_col, "sample_weight"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"Missing OOF columns: {missing}")

    work = oof.copy()
    year = pd.to_numeric(work["oof_year"], errors="coerce")
    years = sorted(int(v) for v in year.dropna().unique())
    if len(years) < 3:
        raise ValueError("Tail-aware experiment requires at least three forward OOF years")

    meta = []
    for val_year in years[1:]:
        fit = work.loc[year < val_year].copy()
        val = work.loc[year == val_year].copy()
        if len(fit) < 80 or val.empty:
            continue
        x_fit, meds = _numeric(fit, features)
        x_val, _ = _numeric(val, features, meds)
        residual = pd.to_numeric(fit[actual_col], errors="coerce") - pd.to_numeric(fit["production_prediction"], errors="coerce")
        weights, _, _ = _tail_weights(fit[actual_col], fit["sample_weight"])
        model = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.02,
            max_depth=3,
            num_leaves=10,
            min_child_samples=50,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=5.0,
            reg_lambda=30.0,
            random_state=seed + val_year,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(x_fit, residual.fillna(0.0).to_numpy(float), sample_weight=weights)
        fit_abs = np.abs(residual.fillna(0.0).to_numpy(float))
        cap = float(np.quantile(fit_abs, 0.95)) if len(fit_abs) else 0.0
        corr = np.clip(np.asarray(model.predict(x_val), dtype=float), -cap, cap)
        meta.append((val, corr))
    if not meta:
        raise ValueError("No valid forward meta-OOF folds for tail-aware layer")

    best = (float("inf"), 0.0)
    for scale in (0.0, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0):
        scores = []
        masses = []
        for val, corr in meta:
            actual = pd.to_numeric(val[actual_col], errors="coerce").to_numpy(float)
            anchor = pd.to_numeric(val["production_prediction"], errors="coerce").to_numpy(float)
            w = pd.to_numeric(val["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
            pred = anchor + scale * corr
            if nonnegative:
                pred = np.maximum(0.0, pred)
            base_m = _weighted_metrics(actual, anchor, w)
            cand_m = _weighted_metrics(actual, pred, w)
            scores.append(_objective(base_m, cand_m))
            masses.append(max(float(np.nansum(w)), 1e-9))
        score = float(np.average(scores, weights=masses))
        if score < best[0]:
            best = (score, float(scale))

    x_fit, medians = _numeric(work, features)
    residual = pd.to_numeric(work[actual_col], errors="coerce") - pd.to_numeric(work["production_prediction"], errors="coerce")
    weights, p90, p95 = _tail_weights(work[actual_col], work["sample_weight"])
    model = LGBMRegressor(
        n_estimators=240,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=10,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=5.0,
        reg_lambda=30.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(x_fit, residual.fillna(0.0).to_numpy(float), sample_weight=weights)
    cap = float(np.quantile(np.abs(residual.fillna(0.0).to_numpy(float)), 0.95))
    return TailAwareLayer(model, list(features), medians, best[1], cap, p90, p95, target)


def apply_tail_aware_layer(layer: TailAwareLayer, frame: pd.DataFrame, anchor: np.ndarray, *, nonnegative: bool) -> np.ndarray:
    score = frame.copy()
    score["production_prediction"] = np.asarray(anchor, dtype=float)
    x, _ = _numeric(score, layer.features, layer.medians)
    correction = layer.scale * np.clip(np.asarray(layer.model.predict(x), dtype=float), -layer.correction_cap, layer.correction_cap)
    pred = np.asarray(anchor, dtype=float) + correction
    return np.maximum(0.0, pred) if nonnegative else pred


def metrics(actual, prediction, weight) -> dict[str, float]:
    return _weighted_metrics(actual, prediction, weight)
