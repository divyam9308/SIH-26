"""Cost R2 experiment: OOF-error-weighted independent squared-error specialist.

The specialist upweights training projects that the current production Cost
stack historically predicted poorly in strict forward OOF. Error-derived
weights, weighting strength, and blend alpha are all frozen from training-only
OOF evidence before the 2022-2025 holdout is evaluated.
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
from backend.app.ml.monthly_training import _fit_pipeline, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_exp105_exp113_baseline import (
    train_window_with_promoted_cost_and_delay as train_current_production,
)

TRAINING_START = 2001
TRAINING_END = 2021
TEST_END = 2025
LAMBDA_GRID = (0.25, 0.5, 1.0, 1.5, 2.0)
ALPHA_GRID = tuple(round(x / 20.0, 2) for x in range(21))
MAX_WEIGHT_MULTIPLIER = 4.0


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


def specialist_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=420,
        learning_rate=0.0225,
        max_depth=4,
        num_leaves=18,
        min_child_samples=45,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=2.0,
        reg_lambda=24.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )


def forward_folds(frame: pd.DataFrame, max_folds: int = 4):
    years_col = pd.to_numeric(frame["completion_year"], errors="coerce")
    years = sorted(int(v) for v in years_col.dropna().unique())
    folds = []
    for year in reversed(years[1:]):
        fitting = frame.loc[years_col < year].copy()
        validation = frame.loc[years_col == year].copy()
        if fitting["canonical_project_id"].nunique() >= 10 and validation["canonical_project_id"].nunique() >= 3:
            folds.append((fitting, validation, year))
        if len(folds) >= max_folds:
            break
    return list(reversed(folds))


def prepare_context(baseline_root: Path) -> dict:
    data, identity = build_training_dataset()
    result = train_current_production(
        TRAINING_START, TRAINING_END, TEST_END,
        data=data, identity=identity, artifact_root=baseline_root,
        verify_frozen_reference=True,
    )
    target = baseline_root / "2001_2021"
    cost_model = joblib.load(target / "cost_model.pkl")
    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, TRAINING_START, TRAINING_END, TEST_END)
    train, test, _ = _build_temporal_delay_priors(train, test)
    cohort = assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
    production_cost = np.asarray(cost_model.predict(cohort), dtype=float)
    return {
        "data": data,
        "identity": identity,
        "train": train,
        "cohort": cohort,
        "cost_model": cost_model,
        "production_cost": production_cost,
        "production_result": result,
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
        raise RuntimeError("Cost production OOF worker was not initialized")
    train_end = int(year) - 1
    with tempfile.TemporaryDirectory(prefix=f"cost-r2-weight-prod-oof-{year}-") as td:
        root = Path(td) / "models"
        train_current_production(
            TRAINING_START, train_end, int(year),
            data=source_data, identity=source_identity, artifact_root=root,
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
    folds = [(validation, int(year)) for _, validation, year in forward_folds(ctx["train"], max_folds) if int(year) - 1 >= 2005]
    if len(folds) < 3:
        raise ValueError("Error-weighted experiment needs at least three strict production OOF folds")
    requested = int(os.environ.get("COST_R2_OOF_WORKERS", "1"))
    workers = max(1, min(requested, len(folds), 4))
    parts, errors = [], []
    if workers == 1:
        for validation, year in folds:
            try:
                parts.append(_strict_production_oof_fold(validation, year, ctx["data"], ctx["identity"]))
                print(f"COST_WEIGHTED_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
            except Exception as exc:
                errors.append(f"{year}: {type(exc).__name__}: {exc}")
    else:
        threads = max(1, min(int(os.environ.get("COST_R2_THREADS_PER_WORKER", "2")), 4))
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
                    print(f"COST_WEIGHTED_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
                except Exception as exc:
                    errors.append(f"{year}: {type(exc).__name__}: {exc}")
    if len(parts) < 3:
        raise ValueError(f"Need >=3 strict production OOF folds; failures={'; '.join(errors) or 'none'}")
    if errors:
        print(f"COST_WEIGHTED_PRODUCTION_OOF_PARTIAL_FAILURES={'; '.join(errors)}", flush=True)
    parts.sort(key=lambda frame: int(frame["oof_year"].iloc[0]))
    return pd.concat(parts, ignore_index=True)


def project_error_scores(oof: pd.DataFrame) -> pd.Series:
    work = oof[["canonical_project_id", "production_residual"]].copy()
    work["squared_error"] = pd.to_numeric(work["production_residual"], errors="coerce") ** 2
    scores = work.groupby("canonical_project_id", dropna=False)["squared_error"].mean()
    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def error_weight_multiplier(frame: pd.DataFrame, prior_oof: pd.DataFrame, strength: float) -> np.ndarray:
    if strength < 0:
        raise ValueError("Weighting strength must be non-negative")
    scores = project_error_scores(prior_oof)
    positive = scores[scores > 0]
    median = float(positive.median()) if not positive.empty else 1.0
    median = max(median, 1e-9)
    normalized = (scores / median).clip(lower=0.0, upper=6.0)
    mapped = frame["canonical_project_id"].map(normalized).fillna(0.0).to_numpy(float)
    multiplier = 1.0 + float(strength) * mapped
    return np.clip(multiplier, 1.0, MAX_WEIGHT_MULTIPLIER)


def build_meta_predictions(ctx: dict, production_oof: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    years_col = pd.to_numeric(production_oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in years_col.dropna().unique())
    if len(years) < 3:
        raise ValueError("Need at least three OOF years for nested error weighting")
    meta_mask = years_col.isin(years[1:]).to_numpy()
    meta = production_oof.loc[meta_mask].copy().reset_index(drop=True)
    predictions = {strength: np.full(len(meta), np.nan, dtype=float) for strength in LAMBDA_GRID}
    unweighted = np.full(len(meta), np.nan, dtype=float)
    cursor = 0
    train_year = pd.to_numeric(ctx["train"]["completion_year"], errors="coerce")
    for year in years[1:]:
        prior_oof = production_oof.loc[years_col < year].copy()
        validation = production_oof.loc[years_col == year].copy()
        fitting = ctx["train"].loc[train_year < year].copy()
        if prior_oof.empty or fitting.empty or validation.empty:
            continue
        n = len(validation)
        base_fit = fitting.copy()
        base_model = _fit_pipeline(specialist_model(19800 + year), base_fit, features, "actual_cost_overrun_percentage")
        unweighted[cursor:cursor + n] = np.asarray(base_model.predict(validation.reindex(columns=features)), dtype=float)
        for strength in LAMBDA_GRID:
            weighted_fit = fitting.copy()
            multiplier = error_weight_multiplier(weighted_fit, prior_oof, strength)
            weighted_fit["sample_weight"] = pd.to_numeric(weighted_fit["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float) * multiplier
            model = _fit_pipeline(specialist_model(19850 + year + int(strength * 100)), weighted_fit, features, "actual_cost_overrun_percentage")
            predictions[strength][cursor:cursor + n] = np.asarray(model.predict(validation.reindex(columns=features)), dtype=float)
        cursor += n
    if cursor != len(meta) or not np.isfinite(unweighted).all() or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Nested weighted specialist OOF predictions are incomplete")
    meta["unweighted_specialist_prediction"] = unweighted
    return meta, predictions


def select_strength_and_alpha(meta: pd.DataFrame, weighted_predictions: dict[float, np.ndarray]) -> dict:
    actual = pd.to_numeric(meta["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(meta["production_prediction"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(meta["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    baseline = weighted_metrics(actual, anchor, weight)
    unweighted = pd.to_numeric(meta["unweighted_specialist_prediction"], errors="coerce").to_numpy(float)
    unweighted_control = weighted_metrics(actual, unweighted, weight)
    eligible, grid = [], []
    for strength, specialist in weighted_predictions.items():
        specialist = np.asarray(specialist, dtype=float)
        specialist_metrics = weighted_metrics(actual, specialist, weight)
        for alpha in ALPHA_GRID:
            blended = (1.0 - alpha) * anchor + alpha * specialist
            metrics = weighted_metrics(actual, blended, weight)
            row = {"strength": float(strength), "alpha": alpha, **metrics}
            grid.append(row)
            if metrics["MAE"] <= baseline["MAE"] + 1e-12:
                eligible.append(row)
    if not eligible:
        raise AssertionError("alpha=0 must satisfy the no-MAE-degradation constraint")
    selected = min(eligible, key=lambda row: (row["RMSE"], row["MAE"], -row["R2"], row["alpha"], row["strength"]))
    return {"baseline": baseline, "unweighted_specialist_control": unweighted_control, "selected": selected, "grid": grid}


def fit_final_weighted_specialist(ctx: dict, production_oof: pd.DataFrame, features: list[str], strength: float):
    fitting = ctx["train"].copy()
    multiplier = error_weight_multiplier(fitting, production_oof, strength)
    fitting["sample_weight"] = pd.to_numeric(fitting["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float) * multiplier
    model = _fit_pipeline(specialist_model(19899), fitting, features, "actual_cost_overrun_percentage")
    return model, multiplier


def run_experiment(baseline_root: Path) -> dict:
    ctx = prepare_context(baseline_root)
    features = list(ctx["cost_model"].features)
    prod_oof = strict_production_oof(ctx, max_folds=4)
    meta, weighted_predictions = build_meta_predictions(ctx, prod_oof, features)
    selection = select_strength_and_alpha(meta, weighted_predictions)
    strength = float(selection["selected"]["strength"])
    alpha = float(selection["selected"]["alpha"])
    specialist, final_multiplier = fit_final_weighted_specialist(ctx, prod_oof, features, strength)
    cohort = ctx["cohort"]
    production = np.asarray(ctx["production_cost"], dtype=float)
    specialist_holdout = np.asarray(specialist.predict(cohort.reindex(columns=features)), dtype=float)
    candidate = (1.0 - alpha) * production + alpha * specialist_holdout
    actual = pd.to_numeric(cohort["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(cohort["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    baseline_metrics = weighted_metrics(actual, production, weight)
    specialist_metrics = weighted_metrics(actual, specialist_holdout, weight)
    candidate_metrics = weighted_metrics(actual, candidate, weight)
    accept = (
        alpha > 0.0
        and strength > 0.0
        and candidate_metrics["MAE"] <= baseline_metrics["MAE"] + 1e-12
        and candidate_metrics["RMSE"] < baseline_metrics["RMSE"]
        and candidate_metrics["R2"] > baseline_metrics["R2"]
    )
    return {
        "experiment_id": "exp_cost_r2_error_weighted_specialist",
        "scope": "cost_only",
        "window": {"training_start": 2001, "training_end": 2021, "test_start": 2022, "test_end": 2025},
        "hypothesis": "upweighting projects with large strict-forward production squared errors can reduce Cost RMSE and raise R2 without sacrificing MAE",
        "selection_policy": "nested training-only OOF: prior folds define project error weights; later OOF folds select weighting strength and blend alpha by minimum RMSE subject to MAE <= production",
        "holdout_used_for_selection": False,
        "full_holdout_retained": True,
        "features_identical_to_current_cost_input_contract": True,
        "selected_weight_strength": strength,
        "selected_alpha": alpha,
        "max_weight_multiplier": MAX_WEIGHT_MULTIPLIER,
        "oof": {
            "strict_production_fold_years": sorted(int(v) for v in pd.to_numeric(prod_oof["oof_year"], errors="coerce").dropna().unique()),
            "selection_rows": int(len(meta)),
            "selection_projects": int(meta["canonical_project_id"].nunique()),
            "production": selection["baseline"],
            "unweighted_l2_specialist_control": selection["unweighted_specialist_control"],
            "selected_weighted_blend": {k: selection["selected"][k] for k in ("strength", "alpha", "MAE", "RMSE", "R2")},
        },
        "final_training_weight_multiplier": {
            "min": float(np.min(final_multiplier)),
            "median": float(np.median(final_multiplier)),
            "max": float(np.max(final_multiplier)),
            "share_above_one": float(np.mean(final_multiplier > 1.0)),
        },
        "holdout": {
            "rows": int(len(cohort)),
            "projects": int(cohort["canonical_project_id"].nunique()),
            "production": baseline_metrics,
            "weighted_specialist_only": specialist_metrics,
            "candidate_blend": candidate_metrics,
            "delta_candidate_minus_production": {k: float(candidate_metrics[k] - baseline_metrics[k]) for k in ("MAE", "RMSE", "R2")},
        },
        "decision": "PROMOTION CANDIDATE" if accept else "REJECT",
        "promotion_allowed": False,
        "acceptance_contract": "A non-zero OOF-selected error weighting and blend must preserve Cost MAE, reduce RMSE, and raise R2 on the unchanged frozen holdout.",
        "production_result": ctx["production_result"].get("lifecycle", {}).get("metrics", {}),
    }
