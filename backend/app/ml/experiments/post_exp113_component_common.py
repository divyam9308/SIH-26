"""Leakage-safe Exp113 component extraction for isolated experiments."""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.post_exp113_delay_common import forward_folds, prepare_context
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay


def decompose(model, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = np.maximum(0.0, np.asarray(model.base_model.predict(frame), dtype=float))
    final = np.maximum(0.0, np.asarray(model.predict(frame), dtype=float))
    return u1, final - u1, final


def component_context() -> dict:
    ctx = prepare_context(2021)
    u1, correction, final = decompose(ctx["delay_model"], ctx["cohort"])
    ctx["production_u1"] = u1
    ctx["exp113_correction"] = correction
    ctx["production_delay"] = final
    return ctx


def component_oof(ctx: dict, max_folds: int = 6) -> pd.DataFrame:
    parts = []
    for _, val, year in forward_folds(ctx["train"], max_folds=max_folds):
        train_end = int(year) - 1
        if train_end < 2005:
            continue
        with tempfile.TemporaryDirectory(prefix=f"exp113-component-{year}-") as td:
            root = Path(td) / "models"
            train_window_with_promoted_cost_and_delay(
                2001, train_end, int(year), data=ctx["full_data"], identity=ctx["identity"],
                artifact_root=root, verify_frozen_reference=False,
            )
            model = joblib.load(root / f"2001_{train_end}" / "delay_model.pkl")
            u1, correction, final = decompose(model, val)
        part = val.copy()
        part["production_u1"] = u1
        part["exp113_correction"] = correction
        part["production_prediction"] = final
        part["residual"] = pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float) - final
        part["oof_year"] = int(year)
        if CALIBRATION_GATE_FEATURE not in part:
            part[CALIBRATION_GATE_FEATURE] = False
        parts.append(part)
        print(f"DELAY_COMPONENT_OOF_FOLD_COMPLETED={year}", flush=True)
    if len(parts) < 3:
        raise ValueError("Need at least three strict forward Exp113 component folds")
    return pd.concat(parts, ignore_index=True)
