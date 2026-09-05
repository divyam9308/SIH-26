"""Cost R2 experiment: blend current production with an independent L2 specialist.

All model-family and blend-weight selection is performed on strict forward
production OOF evidence inside the 2001-2021 training period. The 2022-2025
holdout is touched only once after the configuration is frozen.
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
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

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
ALPHA_GRID = tuple(round(x / 20.0, 2) for x in range(21))


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


def mse_specialists(seed: int = 19700) -> dict[str, object]:
    return {
        "lightgbm_l2": LGBMRegressor(
            objective="regression_l2", n_estimators=360, learning_rate=0.025,
            max_depth=4, num_leaves=18, min_child_samples=45,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=2.0, reg_lambda=20.0,
            random_state=seed, verbosity=-1, n_jobs=1,
        ),
        "xgboost_l2": XGBRegressor(
            objective="reg:squarederror", n_estimators=360, learning_rate=0.025,
            max_depth=4, min_child_weight=8, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=2.0, reg_lambda=20.0, random_state=seed + 1, n_jobs=1,
        ),
        "extra_trees_l2": ExtraTreesRegressor(
            n_estimators=420, criterion="squared_error", min_samples_leaf=3,
            max_features=0.8, random_state=seed + 2, n_jobs=1,
        ),
    }


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
    with tempfile.TemporaryDirectory(prefix=f"cost-r2-prod-oof-{year}-") as td:
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
    if len(folds) < 2:
        raise ValueError("Need at least two strict forward production OOF folds")
    requested = int(os.environ.get("COST_R2_OOF_WORKERS", "1"))
    workers = max(1, min(requested, len(folds), 4))
    parts, errors = [], []
    if workers == 1:
        for validation, year in folds:
            try:
                parts.append(_strict_production_oof_fold(validation, year, ctx["data"], ctx["identity"]))
                print(f"COST_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
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
                    print(f"COST_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
                except Exception as exc:
                    errors.append(f"{year}: {type(exc).__name__}: {exc}")
    if len(parts) < 2:
        raise ValueError(f"Need >=2 strict production OOF folds; failures={'; '.join(errors) or 'none'}")
    if errors:
        print(f"COST_PRODUCTION_OOF_PARTIAL_FAILURES={'; '.join(errors)}", flush=True)
    parts.sort(key=lambda frame: int(frame["oof_year"].iloc[0]))
    return pd.concat(parts, ignore_index=True)


def specialist_oof(ctx: dict, production_oof: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    train_year = pd.to_numeric(ctx["train"]["completion_year"], errors="coerce")
    oof_year = pd.to_numeric(production_oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in oof_year.dropna().unique())
    predictions = {name: np.full(len(production_oof), np.nan, dtype=float) for name in mse_specialists()}
    for year in years:
        fitting = ctx["train"].loc[train_year < year].copy()
        positions = np.flatnonzero(oof_year.to_numpy() == year)
        validation = production_oof.iloc[positions]
        if fitting.empty or validation.empty:
            continue
        for name, estimator in mse_specialists(seed=19700 + year).items():
            model = _fit_pipeline(estimator, fitting, features, "actual_cost_overrun_percentage")
            predictions[name][positions] = np.asarray(model.predict(validation.reindex(columns=features)), dtype=float)
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("MSE specialist OOF predictions are incomplete")
    return predictions


def select_family_and_alpha(production_oof: pd.DataFrame, specialist_predictions: dict[str, np.ndarray]) -> dict:
    actual = pd.to_numeric(production_oof["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(production_oof["production_prediction"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(production_oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    baseline = weighted_metrics(actual, anchor, weight)
    eligible = []
    grid = []
    for family, specialist in specialist_predictions.items():
        specialist = np.asarray(specialist, dtype=float)
        specialist_metrics = weighted_metrics(actual, specialist, weight)
        for alpha in ALPHA_GRID:
            blended = (1.0 - alpha) * anchor + alpha * specialist
            metrics = weighted_metrics(actual, blended, weight)
            row = {"family": family, "alpha": alpha, **metrics}
            grid.append(row)
            if metrics["MAE"] <= baseline["MAE"] + 1e-12:
                eligible.append(row)
    if not eligible:
        raise AssertionError("alpha=0 must always satisfy the no-MAE-degradation constraint")
    selected = min(eligible, key=lambda row: (row["RMSE"], row["MAE"], -row["R2"], row["alpha"], row["family"]))
    return {"baseline": baseline, "selected": selected, "grid": grid}


def fit_final_specialist(ctx: dict, features: list[str], family: str):
    estimators = mse_specialists(seed=19791)
    if family not in estimators:
        raise KeyError(f"Unknown specialist family: {family}")
    return _fit_pipeline(estimators[family], ctx["train"], features, "actual_cost_overrun_percentage")


def run_experiment(baseline_root: Path) -> dict:
    ctx = prepare_context(baseline_root)
    features = list(ctx["cost_model"].features)
    prod_oof = strict_production_oof(ctx, max_folds=4)
    specialist_predictions = specialist_oof(ctx, prod_oof, features)
    selection = select_family_and_alpha(prod_oof, specialist_predictions)
    family = str(selection["selected"]["family"])
    alpha = float(selection["selected"]["alpha"])
    specialist = fit_final_specialist(ctx, features, family)
    specialist_holdout = np.asarray(specialist.predict(ctx["cohort"].reindex(columns=features)), dtype=float)
    production = np.asarray(ctx["production_cost"], dtype=float)
    candidate = (1.0 - alpha) * production + alpha * specialist_holdout
    cohort = ctx["cohort"]
    actual = pd.to_numeric(cohort["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(cohort["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    baseline_metrics = weighted_metrics(actual, production, weight)
    specialist_metrics = weighted_metrics(actual, specialist_holdout, weight)
    candidate_metrics = weighted_metrics(actual, candidate, weight)
    accept = (
        candidate_metrics["MAE"] <= baseline_metrics["MAE"] + 1e-12
        and candidate_metrics["RMSE"] < baseline_metrics["RMSE"]
        and candidate_metrics["R2"] > baseline_metrics["R2"]
    )
    return {
        "experiment_id": "exp_cost_r2_mse_oof_blend",
        "scope": "cost_only",
        "window": {"training_start": 2001, "training_end": 2021, "test_start": 2022, "test_end": 2025},
        "hypothesis": "an independent squared-error specialist can reduce Cost RMSE/R2 loss while an OOF-selected convex blend preserves production MAE",
        "selection_policy": "model family and alpha selected only from strict forward production OOF within 2001-2021; minimize RMSE subject to MAE <= production OOF MAE",
        "holdout_used_for_selection": False,
        "full_holdout_retained": True,
        "features_identical_to_current_cost_input_contract": True,
        "selected_family": family,
        "selected_alpha": alpha,
        "oof": {
            "fold_years": sorted(int(v) for v in pd.to_numeric(prod_oof["oof_year"], errors="coerce").dropna().unique()),
            "rows": int(len(prod_oof)),
            "projects": int(prod_oof["canonical_project_id"].nunique()),
            "production": selection["baseline"],
            "selected_blend": {k: selection["selected"][k] for k in ("family", "alpha", "MAE", "RMSE", "R2")},
        },
        "holdout": {
            "rows": int(len(cohort)),
            "projects": int(cohort["canonical_project_id"].nunique()),
            "production": baseline_metrics,
            "specialist_only": specialist_metrics,
            "candidate_blend": candidate_metrics,
            "delta_candidate_minus_production": {k: float(candidate_metrics[k] - baseline_metrics[k]) for k in ("MAE", "RMSE", "R2")},
        },
        "decision": "PROMOTION CANDIDATE" if accept else "REJECT",
        "promotion_allowed": False,
        "acceptance_contract": "Cost MAE must not worsen; Cost RMSE must fall; Cost R2 must rise on the unchanged frozen holdout.",
        "production_result": ctx["production_result"].get("lifecycle", {}).get("metrics", {}),
    }
