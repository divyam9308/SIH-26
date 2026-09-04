"""Performance-preserving execution wrapper for canonical Exp105 Cost + Exp113 Delay.

This module does not change model features, folds, estimators, targets, calibration,
or promotion criteria. It only parallelizes independent canonical OOF work and lets
LightGBM use the configured worker layout. The canonical trainer remains the single
source of model logic and artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import threading
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor

from backend.app.ml import production_exp105_exp113_baseline as canonical

_PATCH_LOCK = threading.Lock()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def worker_layout() -> dict[str, int]:
    cpu = max(1, os.cpu_count() or 1)
    fold_default = 2 if cpu >= 4 else 1
    fold_jobs = min(_env_int("SIH_FOLD_JOBS", fold_default), cpu)
    model_default = max(1, cpu // fold_jobs)
    model_threads = min(_env_int("SIH_MODEL_THREADS", model_default), cpu)
    return {"cpu_count": cpu, "fold_jobs": fold_jobs, "model_threads": model_threads}


def _threaded_lgbm(*args, **kwargs):
    kwargs["n_jobs"] = worker_layout()["model_threads"]
    return LGBMRegressor(*args, **kwargs)


def _cost_fold(fitting: pd.DataFrame, validation: pd.DataFrame, year: int, features: list[str], family: str):
    inner = canonical._raw_cost_oof(fitting, features, family, 3)
    if inner.empty:
        return None
    calibration = canonical.shrunk_calibration(inner, 40.0)
    model = canonical._fit_pipeline(
        canonical._regressors(canonical.PRODUCTION_COST_SEED)[family],
        fitting,
        features,
        "actual_cost_overrun_percentage",
    )
    raw = np.asarray(model.predict(validation.reindex(columns=features)), dtype=float)
    prediction = raw + canonical._corrections(validation, raw, calibration)
    part = validation.copy()
    part["production_prediction"] = prediction
    part["residual"] = (
        pd.to_numeric(part["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
        - prediction
    )
    part["oof_year"] = int(year)
    return part


def _parallel_current_cost_oof(train: pd.DataFrame, production_model) -> pd.DataFrame:
    features = list(production_model.features)
    family = canonical._family(production_model)
    folds = canonical._forward_folds(train, 6)
    jobs = worker_layout()["fold_jobs"]
    parts = Parallel(n_jobs=jobs, prefer="threads", require="sharedmem")(
        delayed(_cost_fold)(fitting, validation, year, features, family)
        for fitting, validation, year in folds
    )
    out = [part for part in parts if part is not None]
    if len(out) < 2:
        raise ValueError("Exp105 production promotion requires at least two forward Cost OOF folds")
    return pd.concat(out, ignore_index=True)


def _delay_fold(fitting: pd.DataFrame, validation: pd.DataFrame, year: int, features: list[str], base):
    models = canonical._fit_aft_family_models(fitting, features)
    remaining = canonical._aft_remaining_prediction(models, base.weights, validation, features)
    raw = canonical._delay_from_remaining(validation, remaining)
    prediction = np.maximum(0.0, raw + canonical._corrections(validation, raw, base.calibration))
    part = validation.copy()
    part["base_prediction"] = prediction
    part["production_prediction"] = prediction
    part["residual"] = pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float) - prediction
    part["oof_year"] = int(year)
    return part


def _parallel_base_delay_oof(train: pd.DataFrame, u1_model) -> pd.DataFrame:
    if not hasattr(u1_model, "base_model"):
        raise TypeError("Exp113 production promotion requires the U1 Delay wrapper")
    base = u1_model.base_model
    features = list(base.model_features)
    train_delay = canonical._remaining_frame(train)
    folds = canonical._forward_folds(train_delay, 8)
    jobs = worker_layout()["fold_jobs"]
    parts = Parallel(n_jobs=jobs, prefer="threads", require="sharedmem")(
        delayed(_delay_fold)(fitting, validation, year, features, base)
        for fitting, validation, year in folds
    )
    if len(parts) < 4:
        raise ValueError("Exp113 production promotion requires at least four base Delay OOF folds")
    return pd.concat(parts, ignore_index=True)


@contextmanager
def _performance_patch():
    with _PATCH_LOCK:
        old_lgbm = canonical.LGBMRegressor
        old_cost = canonical._current_cost_oof
        old_delay = canonical._base_delay_oof
        canonical.LGBMRegressor = _threaded_lgbm
        canonical._current_cost_oof = _parallel_current_cost_oof
        canonical._base_delay_oof = _parallel_base_delay_oof
        try:
            yield
        finally:
            canonical.LGBMRegressor = old_lgbm
            canonical._current_cost_oof = old_cost
            canonical._base_delay_oof = old_delay


def train_window_with_promoted_cost_and_delay(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root=None,
) -> dict:
    started = time.perf_counter()
    with _performance_patch():
        result = canonical.train_window_with_promoted_cost_and_delay(
            training_start,
            training_end,
            test_end,
            data=data,
            identity=identity,
            artifact_root=artifact_root,
        )
    result.setdefault("metadata", {})["training_performance"] = {
        **worker_layout(),
        "elapsed_seconds": round(float(time.perf_counter() - started), 3),
        "model_logic": "canonical_exp105_exp113_unchanged",
    }
    return result


Exp105CostProductionModel = canonical.Exp105CostProductionModel
Exp113DelayProductionModel = canonical.Exp113DelayProductionModel
PRODUCTION_COST_BASELINE = canonical.PRODUCTION_COST_BASELINE
PRODUCTION_DELAY_BASELINE = canonical.PRODUCTION_DELAY_BASELINE
