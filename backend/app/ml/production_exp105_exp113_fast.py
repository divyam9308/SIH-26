"""Performance-preserving execution wrapper for canonical Exp105 Cost + Exp113 Delay.

This module does not change model features, folds, estimators, targets, calibration,
or promotion criteria. It only accelerates independent work and persists deterministic
OOF intermediates keyed by the full training frame. The canonical trainer remains the
single source of model logic and artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor

from backend.app.ml.monthly_training import MODEL_ROOT
from backend.app.ml import production_exp105_exp113_baseline as canonical

CACHE_VERSION = "exp105-exp113-fast-v1"
_PATCH_LOCK = threading.Lock()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def worker_layout() -> dict[str, int]:
    """Return a conservative layout that avoids CPU oversubscription on laptops/CI."""
    cpu = max(1, os.cpu_count() or 1)
    fold_default = 2 if cpu >= 4 else 1
    fold_jobs = min(_env_int("SIH_FOLD_JOBS", fold_default), cpu)
    model_default = max(1, cpu // fold_jobs)
    model_threads = min(_env_int("SIH_MODEL_THREADS", model_default), cpu)
    return {"cpu_count": cpu, "fold_jobs": fold_jobs, "model_threads": model_threads}


def _cache_root() -> Path:
    configured = os.getenv("SIH_TRAIN_CACHE_DIR")
    root = Path(configured).expanduser() if configured else MODEL_ROOT / ".training_cache"
    target = root / CACHE_VERSION
    target.mkdir(parents=True, exist_ok=True)
    return target


def _frame_key(kind: str, frame: pd.DataFrame) -> str:
    # joblib.hash is deterministic for pandas objects and includes values, dtypes,
    # index and column order. CACHE_VERSION explicitly invalidates code-level changes.
    return joblib.hash((CACHE_VERSION, kind, frame), hash_name="sha1")


def _load_cached_frame(kind: str, frame: pd.DataFrame) -> pd.DataFrame | None:
    if os.getenv("SIH_DISABLE_TRAIN_CACHE", "0") == "1":
        return None
    path = _cache_root() / f"{kind}-{_frame_key(kind, frame)}.joblib"
    if not path.exists():
        return None
    try:
        cached = joblib.load(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None
    return cached if isinstance(cached, pd.DataFrame) else None


def _store_cached_frame(kind: str, source: pd.DataFrame, value: pd.DataFrame) -> None:
    if os.getenv("SIH_DISABLE_TRAIN_CACHE", "0") == "1":
        return
    path = _cache_root() / f"{kind}-{_frame_key(kind, source)}.joblib"
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    joblib.dump(value, tmp, compress=1)
    tmp.replace(path)


def _threaded_lgbm(*args, **kwargs):
    # Canonical code currently passes n_jobs=1 explicitly. Override only the
    # execution resource count; all statistical/model parameters remain unchanged.
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
    cached = _load_cached_frame("cost-oof", train)
    if cached is not None:
        _RUNTIME_STATE["cost_oof_cache_hit"] = True
        return cached.copy()

    features = list(production_model.features)
    family = canonical._family(production_model)
    folds = canonical._forward_folds(train, 6)
    jobs = worker_layout()["fold_jobs"]
    parts = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_cost_fold)(fitting, validation, year, features, family)
        for fitting, validation, year in folds
    )
    out = [part for part in parts if part is not None]
    if len(out) < 2:
        raise ValueError("Exp105 production promotion requires at least two forward Cost OOF folds")
    result = pd.concat(out, ignore_index=True)
    _store_cached_frame("cost-oof", train, result)
    return result


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
    cached = _load_cached_frame("delay-base-oof", train)
    if cached is not None:
        _RUNTIME_STATE["delay_oof_cache_hit"] = True
        return cached.copy()
    if not hasattr(u1_model, "base_model"):
        raise TypeError("Exp113 production promotion requires the U1 Delay wrapper")
    base = u1_model.base_model
    features = list(base.model_features)
    train_delay = canonical._remaining_frame(train)
    folds = canonical._forward_folds(train_delay, 8)
    jobs = worker_layout()["fold_jobs"]
    parts = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(_delay_fold)(fitting, validation, year, features, base)
        for fitting, validation, year in folds
    )
    if len(parts) < 4:
        raise ValueError("Exp113 production promotion requires at least four base Delay OOF folds")
    result = pd.concat(parts, ignore_index=True)
    _store_cached_frame("delay-base-oof", train, result)
    return result


_RUNTIME_STATE: dict[str, object] = {}


@contextmanager
def _performance_patch():
    """Temporarily replace only execution primitives used by the canonical trainer."""
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
    artifact_root: Path | None = None,
    verify_frozen_reference: bool = True,
) -> dict:
    """Run the canonical trainer with performance-only execution substitutions."""
    global _RUNTIME_STATE
    _RUNTIME_STATE = {"cost_oof_cache_hit": False, "delay_oof_cache_hit": False}
    started = time.perf_counter()
    with _performance_patch():
        result = canonical.train_window_with_promoted_cost_and_delay(
            training_start,
            training_end,
            test_end,
            data=data,
            identity=identity,
            artifact_root=artifact_root,
            verify_frozen_reference=verify_frozen_reference,
        )
    elapsed = time.perf_counter() - started
    performance = {
        **worker_layout(),
        **_RUNTIME_STATE,
        "elapsed_seconds": round(float(elapsed), 3),
        "cache_version": CACHE_VERSION,
        "model_logic": "canonical_exp105_exp113_unchanged",
    }
    result["performance"] = performance
    result.setdefault("metadata", {})["training_performance"] = performance
    return result


# Re-export wrappers/classes used by callers and tests without duplicating model logic.
Exp105CostProductionModel = canonical.Exp105CostProductionModel
Exp113DelayProductionModel = canonical.Exp113DelayProductionModel
PRODUCTION_COST_BASELINE = canonical.PRODUCTION_COST_BASELINE
PRODUCTION_DELAY_BASELINE = canonical.PRODUCTION_DELAY_BASELINE
