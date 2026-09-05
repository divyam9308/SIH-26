"""Exp94: production-anchored Cost correction from model-family disagreement.

This is a batch experiment only. It intentionally does not register an interactive
adapter. The current Exp105 Cost + Exp113 Delay production stack is freshly
retrained for every evaluation window and again for strict forward production OOF
folds. ExtraTrees, LightGBM and XGBoost Cost predictions are then used only as
training-time uncertainty/meta signals for a small residual correction.

Selection is nested and leakage-safe:
1. Current production predictions are generated strictly forward OOF.
2. Family predictions for those rows are fit using projects completed before the
   corresponding OOF year.
3. The residual corrector is itself evaluated in a second forward meta-OOF layer.
4. Correction scale is chosen only from that meta-OOF evidence, with zero
   correction always available.
5. The future holdout is evaluated once after every choice is frozen.
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights, build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED, _production_cost_evaluation_rows
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_exp105_exp113_baseline import (
    PRODUCTION_COST_BASELINE,
    PRODUCTION_DELAY_BASELINE,
    train_window_with_promoted_cost_and_delay as train_current_production,
)

EXPERIMENT_ID = "exp_94"
EXPERIMENT_NAME = "Cost model-family disagreement booster"
EXPERIMENT_SCOPE = "cost"
PROMOTION_ALLOWED = False
TRAINING_START = 2001
TEST_END = 2025
ALLOWED_TRAINING_ENDS = (2019, 2021)
FAMILIES = ("extra_trees", "lightgbm", "xgboost")
SCALE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
META_FEATURES = (
    "production_prediction",
    "exp94_extra_trees",
    "exp94_lightgbm",
    "exp94_xgboost",
    "exp94_family_mean",
    "exp94_family_std",
    "exp94_family_range",
    "duration_ratio",
    "cost_escalation_percentage",
)


def window_contract(training_end: int) -> tuple[int, int]:
    training_end = int(training_end)
    if training_end not in ALLOWED_TRAINING_ENDS:
        raise ValueError(f"Exp94 allows training_end only in {ALLOWED_TRAINING_ENDS}")
    return training_end + 1, TEST_END


def weighted_metrics(actual, prediction, weight) -> dict[str, float]:
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


def forward_folds(frame: pd.DataFrame, max_folds: int = 4):
    completion_year = pd.to_numeric(frame["completion_year"], errors="coerce")
    years = sorted(int(v) for v in completion_year.dropna().unique())
    folds = []
    for year in reversed(years[1:]):
        fitting = frame.loc[completion_year < year].copy()
        validation = frame.loc[completion_year == year].copy()
        if fitting["canonical_project_id"].nunique() >= 10 and validation["canonical_project_id"].nunique() >= 3:
            folds.append((fitting, validation, year))
        if len(folds) >= max_folds:
            break
    return list(reversed(folds))


def prepare_context(training_end: int, baseline_root: Path) -> dict:
    test_start, test_end = window_contract(training_end)
    data, identity = build_training_dataset()
    result = train_current_production(
        TRAINING_START,
        int(training_end),
        test_end,
        data=data,
        identity=identity,
        artifact_root=baseline_root,
    )
    target = baseline_root / f"2001_{int(training_end)}"
    cost_model = joblib.load(target / "cost_model.pkl")

    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, TRAINING_START, int(training_end), test_end)
    train, test, _ = _build_temporal_delay_priors(train, test)
    cohort = assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
    production_cost = np.asarray(cost_model.predict(cohort), dtype=float)

    completion_year = pd.to_numeric(cohort["completion_year"], errors="coerce")
    if completion_year.notna().any() and int(completion_year.min()) < test_start:
        raise AssertionError("Exp94 evaluation cohort contains pre-holdout completion years")

    return {
        "data": data,
        "identity": identity,
        "train": train,
        "cohort": cohort,
        "cost_model": cost_model,
        "production_cost": production_cost,
        "production_result": result,
        "test_start": test_start,
        "test_end": test_end,
    }


_OOF_DATA = None
_OOF_IDENTITY = None


def _init_worker(data, identity):
    global _OOF_DATA, _OOF_IDENTITY
    _OOF_DATA, _OOF_IDENTITY = data, identity


def _strict_production_oof_fold(validation: pd.DataFrame, year: int, data=None, identity=None) -> pd.DataFrame:
    source_data = _OOF_DATA if data is None else data
    source_identity = _OOF_IDENTITY if identity is None else identity
    if source_data is None or source_identity is None:
        raise RuntimeError("Exp94 production OOF worker was not initialized")
    train_end = int(year) - 1
    with tempfile.TemporaryDirectory(prefix=f"exp94-prod-oof-{year}-") as td:
        root = Path(td) / "models"
        train_current_production(
            TRAINING_START,
            train_end,
            int(year),
            data=source_data,
            identity=source_identity,
            artifact_root=root,
            verify_frozen_reference=False,
        )
        model = joblib.load(root / f"2001_{train_end}" / "cost_model.pkl")
        prediction = np.asarray(model.predict(validation), dtype=float)
    part = validation.copy()
    part["production_prediction"] = prediction
    part["production_residual"] = (
        pd.to_numeric(part["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float) - prediction
    )
    part["oof_year"] = int(year)
    return part


def strict_production_oof(ctx: dict, max_folds: int = 4) -> pd.DataFrame:
    folds = [
        (validation, int(year))
        for _, validation, year in forward_folds(ctx["train"], max_folds)
        if int(year) - 1 >= 2005
    ]
    if len(folds) < 3:
        raise ValueError("Exp94 needs at least three strict forward production OOF folds")

    requested = int(os.environ.get("EXP94_OOF_WORKERS", "1"))
    workers = max(1, min(requested, len(folds), 4))
    parts: list[pd.DataFrame] = []
    errors: list[str] = []

    if workers == 1:
        for validation, year in folds:
            try:
                parts.append(_strict_production_oof_fold(validation, year, ctx["data"], ctx["identity"]))
                print(f"EXP94_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
            except Exception as exc:
                errors.append(f"{year}: {type(exc).__name__}: {exc}")
    else:
        threads = max(1, min(int(os.environ.get("EXP94_THREADS_PER_WORKER", "2")), 4))
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(threads)
        os.environ["LOKY_MAX_CPU_COUNT"] = str(threads)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(ctx["data"], ctx["identity"]),
        ) as pool:
            pending = {pool.submit(_strict_production_oof_fold, validation, year): year for validation, year in folds}
            for future in as_completed(pending):
                year = pending[future]
                try:
                    parts.append(future.result())
                    print(f"EXP94_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
                except Exception as exc:
                    errors.append(f"{year}: {type(exc).__name__}: {exc}")

    if len(parts) < 3:
        raise ValueError(f"Exp94 needs >=3 production OOF folds; failures={'; '.join(errors) or 'none'}")
    if errors:
        print(f"EXP94_PRODUCTION_OOF_PARTIAL_FAILURES={'; '.join(errors)}", flush=True)
    parts.sort(key=lambda frame: int(frame["oof_year"].iloc[0]))
    return pd.concat(parts, ignore_index=True)


def _add_disagreement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns = [f"exp94_{family}" for family in FAMILIES]
    values = result[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    result["exp94_family_mean"] = np.nanmean(values, axis=1)
    result["exp94_family_std"] = np.nanstd(values, axis=1)
    result["exp94_family_range"] = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    return result


def family_oof_features(ctx: dict, production_oof: pd.DataFrame) -> pd.DataFrame:
    result = production_oof.copy()
    features = list(ctx["cost_model"].features)
    training_year = pd.to_numeric(ctx["train"]["completion_year"], errors="coerce")
    oof_year = pd.to_numeric(result["oof_year"], errors="coerce")
    for family in FAMILIES:
        result[f"exp94_{family}"] = np.nan

    for year in sorted(int(v) for v in oof_year.dropna().unique()):
        fitting = ctx["train"].loc[training_year < year].copy()
        positions = np.flatnonzero(oof_year.to_numpy() == year)
        validation = result.iloc[positions]
        if fitting.empty or validation.empty:
            continue
        for family in FAMILIES:
            model = _fit_pipeline(
                _regressors(PRODUCTION_COST_SEED)[family],
                fitting,
                features,
                "actual_cost_overrun_percentage",
            )
            result.iloc[positions, result.columns.get_loc(f"exp94_{family}")] = np.asarray(
                model.predict(validation.reindex(columns=features)), dtype=float
            )

    family_columns = [f"exp94_{family}" for family in FAMILIES]
    if not np.isfinite(result[family_columns].to_numpy(dtype=float)).all():
        raise ValueError("Exp94 family OOF predictions are incomplete")
    return _add_disagreement_columns(result)


def family_holdout_features(ctx: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    score = cohort.copy()
    features = list(ctx["cost_model"].features)
    for family in FAMILIES:
        model = _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[family],
            ctx["train"],
            features,
            "actual_cost_overrun_percentage",
        )
        score[f"exp94_{family}"] = np.asarray(model.predict(score.reindex(columns=features)), dtype=float)
    return _add_disagreement_columns(score)


def _numeric_design(train: pd.DataFrame, score: pd.DataFrame, features=META_FEATURES):
    medians: dict[str, float] = {}
    left: dict[str, pd.Series] = {}
    right: dict[str, pd.Series] = {}
    for col in features:
        a = pd.to_numeric(train.get(col, pd.Series(np.nan, index=train.index)), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        b = pd.to_numeric(score.get(col, pd.Series(np.nan, index=score.index)), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = float(a.median()) if a.notna().any() else 0.0
        medians[col] = median
        left[col] = a.fillna(median)
        right[col] = b.fillna(median)
    return pd.DataFrame(left, index=train.index), pd.DataFrame(right, index=score.index), medians


def _residual_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=220,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=12,
        min_child_samples=45,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=3.0,
        reg_lambda=30.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )


def _fit_residual_and_predict(fitting: pd.DataFrame, validation: pd.DataFrame, seed: int) -> tuple[np.ndarray, float]:
    x_fit, x_val, _ = _numeric_design(fitting, validation)
    target = pd.to_numeric(fitting["production_residual"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(fitting["sample_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = np.isfinite(target) & np.isfinite(weights) & (weights >= 0)
    if int(mask.sum()) < 20:
        raise ValueError("Insufficient finite Exp94 residual meta-training rows")
    model = _residual_model(seed)
    model.fit(x_fit.loc[mask], target[mask], sample_weight=weights[mask])
    raw = np.asarray(model.predict(x_val), dtype=float)
    cap = _weighted_quantile(np.abs(target[mask]), weights[mask], 0.90)
    return np.clip(raw, -cap, cap), float(cap)


def meta_oof_corrections(frame: pd.DataFrame) -> pd.DataFrame:
    oof_year = pd.to_numeric(frame["oof_year"], errors="coerce")
    parts = []
    for year in sorted(int(v) for v in oof_year.dropna().unique()):
        fitting = frame.loc[oof_year < year].copy()
        validation = frame.loc[oof_year == year].copy()
        if fitting["canonical_project_id"].nunique() < 10 or validation["canonical_project_id"].nunique() < 3:
            continue
        correction, cap = _fit_residual_and_predict(fitting, validation, 9400 + year)
        part = validation.copy()
        part["exp94_raw_correction"] = correction
        part["exp94_correction_cap"] = cap
        part["meta_validation_year"] = year
        parts.append(part)
    if len(parts) < 2:
        raise ValueError("Exp94 needs at least two forward meta-OOF correction folds")
    return pd.concat(parts, ignore_index=True)


def select_scale(meta_oof: pd.DataFrame) -> dict:
    actual = pd.to_numeric(meta_oof["actual_cost_overrun_percentage"], errors="coerce").to_numpy(dtype=float)
    production = pd.to_numeric(meta_oof["production_prediction"], errors="coerce").to_numpy(dtype=float)
    correction = pd.to_numeric(meta_oof["exp94_raw_correction"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(meta_oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    baseline = weighted_metrics(actual, production, weights)
    grid = []
    for scale in SCALE_GRID:
        candidate = production + float(scale) * correction
        metrics = weighted_metrics(actual, candidate, weights)
        grid.append({"scale": float(scale), **metrics})
    selected = min(grid, key=lambda row: (row["MAE"], row["RMSE"], row["scale"]))
    return {"baseline": baseline, "selected": selected, "grid": grid}


def final_correction(oof: pd.DataFrame, holdout: pd.DataFrame) -> tuple[np.ndarray, float]:
    return _fit_residual_and_predict(oof, holdout, 9494)


def run_experiment(training_end: int, baseline_root: Path) -> dict:
    training_end = int(training_end)
    test_start, test_end = window_contract(training_end)
    ctx = prepare_context(training_end, baseline_root)

    production_oof = strict_production_oof(ctx, max_folds=4)
    enriched_oof = family_oof_features(ctx, production_oof)
    meta_oof = meta_oof_corrections(enriched_oof)
    selection = select_scale(meta_oof)
    scale = float(selection["selected"]["scale"])

    holdout = ctx["cohort"].copy()
    holdout["production_prediction"] = np.asarray(ctx["production_cost"], dtype=float)
    enriched_holdout = family_holdout_features(ctx, holdout)
    raw_correction, final_cap = final_correction(enriched_oof, enriched_holdout)
    production = np.asarray(ctx["production_cost"], dtype=float)
    candidate = production + scale * raw_correction

    actual = pd.to_numeric(enriched_holdout["actual_cost_overrun_percentage"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(enriched_holdout["sample_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    production_metrics = weighted_metrics(actual, production, weights)
    candidate_metrics = weighted_metrics(actual, candidate, weights)
    improvement_pct = (
        (production_metrics["MAE"] - candidate_metrics["MAE"]) / production_metrics["MAE"] * 100.0
        if production_metrics["MAE"]
        else 0.0
    )

    metadata = dict(ctx["production_result"].get("metadata") or {})
    lifecycle_metrics = dict(ctx["production_result"].get("lifecycle", {}).get("metrics") or {})
    delay_metrics = dict(lifecycle_metrics.get("delay") or {})
    scientific_improvement = candidate_metrics["MAE"] < production_metrics["MAE"] - 1e-12

    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "scope": EXPERIMENT_SCOPE,
        "window": {
            "training_start": TRAINING_START,
            "training_end": training_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "production_cost_baseline": metadata.get("production_cost_baseline", PRODUCTION_COST_BASELINE),
        "production_delay_baseline": metadata.get("production_delay_baseline", PRODUCTION_DELAY_BASELINE),
        "comparison_projects": int(enriched_holdout["canonical_project_id"].nunique()),
        "comparison_snapshots": int(len(enriched_holdout)),
        "full_holdout_retained": True,
        "holdout_used_for_selection": False,
        "delay_predictions_identical": True,
        "promotion_allowed": PROMOTION_ALLOWED,
        "family_models": list(FAMILIES),
        "meta_features": list(META_FEATURES),
        "production_oof": {
            "fold_years": sorted(int(v) for v in pd.to_numeric(enriched_oof["oof_year"], errors="coerce").dropna().unique()),
            "rows": int(len(enriched_oof)),
            "projects": int(enriched_oof["canonical_project_id"].nunique()),
        },
        "meta_oof": {
            "fold_years": sorted(int(v) for v in pd.to_numeric(meta_oof["meta_validation_year"], errors="coerce").dropna().unique()),
            "rows": int(len(meta_oof)),
            "projects": int(meta_oof["canonical_project_id"].nunique()),
            "production": selection["baseline"],
            "selected": selection["selected"],
            "scale_grid": selection["grid"],
        },
        "selected_correction_scale": scale,
        "final_training_correction_cap": final_cap,
        "family_disagreement_holdout": {
            "mean_std": float(pd.to_numeric(enriched_holdout["exp94_family_std"], errors="coerce").mean()),
            "mean_range": float(pd.to_numeric(enriched_holdout["exp94_family_range"], errors="coerce").mean()),
        },
        "cost": {
            "production": production_metrics,
            "experiment": candidate_metrics,
            "mae_improvement_pct": float(improvement_pct),
        },
        "delay": {
            "production": delay_metrics,
            "experiment": delay_metrics,
            "identical_by_contract": True,
        },
        "execution_verdict": "EXECUTION VALID",
        "scientific_verdict": "IMPROVEMENT" if scientific_improvement else "DO NOT PROMOTE",
        "selection_contract": "scale chosen only from second-level forward meta-OOF MAE; zero correction is always selectable",
    }
