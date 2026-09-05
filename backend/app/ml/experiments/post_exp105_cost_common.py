"""Shared leakage-safe harness for isolated post-Exp105 Cost experiments.

This module never promotes artifacts. It rebuilds the canonical Exp105+Exp113
production stack only inside temporary directories, produces strict forward OOF
Cost predictions, and evaluates candidates on the unchanged project-balanced
holdout cohort.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights, build_training_dataset
from backend.app.ml.monthly_training import _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay


def metric(frame: pd.DataFrame, actual: str, prediction: np.ndarray) -> float:
    return float(_regression_metrics(frame[actual], prediction, frame["sample_weight"], frame["canonical_project_id"])["MAE"])


def regression_metrics(frame: pd.DataFrame, actual: str, prediction: np.ndarray) -> dict:
    return _regression_metrics(frame[actual], prediction, frame["sample_weight"], frame["canonical_project_id"])


def numeric_design(train: pd.DataFrame, score: pd.DataFrame, features: list[str]):
    cols = [c for c in features if c in train.columns and c in score.columns]
    medians: dict[str, float] = {}
    a: dict[str, pd.Series] = {}
    b: dict[str, pd.Series] = {}
    for col in cols:
        x = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(score[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(x.median()) if x.notna().any() else 0.0
        medians[col] = median
        a[col] = x.fillna(median)
        b[col] = y.fillna(median)
        # Availability itself is predictive and must be derived from as-of inputs only.
        a[f"{col}__missing"] = x.isna().astype(float)
        b[f"{col}__missing"] = y.isna().astype(float)
    return list(a), medians, pd.DataFrame(a, index=train.index), pd.DataFrame(b, index=score.index)


def forward_folds(frame: pd.DataFrame, max_folds: int = 6):
    years = pd.to_numeric(frame["completion_year"], errors="coerce")
    values = sorted(int(v) for v in years.dropna().unique())
    folds = []
    for year in reversed(values[1:]):
        fit = frame.loc[years < year].copy()
        val = frame.loc[years == year].copy()
        if fit["canonical_project_id"].nunique() >= 10 and val["canonical_project_id"].nunique() >= 3:
            folds.append((fit, val, year))
        if len(folds) >= max_folds:
            break
    return list(reversed(folds))


def _decompose(model, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(model.base_model.predict(frame), dtype=float)
    final = np.asarray(model.predict(frame), dtype=float)
    return base, final - base, final


def prepare_context(training_end: int = 2021, test_end: int = 2025) -> dict:
    data, identity = build_training_dataset()
    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, 2001, training_end, test_end)
    cohort = assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
    with tempfile.TemporaryDirectory(prefix=f"post-exp105-{training_end}-") as td:
        root = Path(td) / "models"
        train_window_with_promoted_cost_and_delay(
            2001, training_end, test_end, data=data, identity=identity,
            artifact_root=root, verify_frozen_reference=(training_end == 2021 and test_end == 2025),
        )
        model = joblib.load(root / f"2001_{training_end}" / "cost_model.pkl")
        base, correction, production = _decompose(model, cohort)
    return {
        "training_end": training_end,
        "test_end": test_end,
        "data": data,
        "identity": identity,
        "train": train,
        "cohort": cohort,
        "production_model": model,
        "production_base": base,
        "production_correction": correction,
        "production_prediction": production,
    }


def production_oof(ctx: dict, max_folds: int = 6) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, val, year in forward_folds(ctx["train"], max_folds=max_folds):
        train_end = int(year) - 1
        if train_end < 2005:
            continue
        with tempfile.TemporaryDirectory(prefix=f"exp105-oof-{year}-") as td:
            root = Path(td) / "models"
            train_window_with_promoted_cost_and_delay(
                2001, train_end, int(year), data=ctx["data"], identity=ctx["identity"],
                artifact_root=root, verify_frozen_reference=False,
            )
            model = joblib.load(root / f"2001_{train_end}" / "cost_model.pkl")
            base, correction, prediction = _decompose(model, val)
        part = val.copy()
        part["production_base"] = base
        part["exp105_correction"] = correction
        part["production_prediction"] = prediction
        part["residual"] = pd.to_numeric(part["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float) - prediction
        part["oof_year"] = int(year)
        parts.append(part)
        print(f"COST_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
    if len(parts) < 3:
        raise ValueError("Need at least three strict forward Cost production OOF folds")
    return pd.concat(parts, ignore_index=True)


def persist(exp_id: str, name: str, ctx: dict, prediction: np.ndarray, details: dict, output: str) -> dict:
    cohort = ctx["cohort"]
    baseline = np.asarray(ctx["production_prediction"], dtype=float)
    candidate = np.asarray(prediction, dtype=float)
    bm = regression_metrics(cohort, "actual_cost_overrun_percentage", baseline)
    cm = regression_metrics(cohort, "actual_cost_overrun_percentage", candidate)
    verdict = "PROMOTION CANDIDATE" if float(cm["MAE"]) < float(bm["MAE"]) else "DO NOT PROMOTE"
    result = {
        "experiment_id": exp_id,
        "experiment_name": name,
        "scope": "cost",
        "training_start": 2001,
        "training_end": ctx["training_end"],
        "test_start": ctx["training_end"] + 1,
        "test_end": ctx["test_end"],
        "comparison_test_projects": int(cohort["canonical_project_id"].nunique()),
        "comparison_test_snapshots": int(len(cohort)),
        "production": bm,
        "candidate": cm,
        "mae_delta": float(cm["MAE"]) - float(bm["MAE"]),
        "holdout_used_for_selection": False,
        "promotion_allowed": False,
        "scientific_verdict": verdict,
        "details": _json_safe(details),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    print(f"{exp_id.upper()}_PRODUCTION_COST_MAE={float(bm['MAE']):.6f}")
    print(f"{exp_id.upper()}_CANDIDATE_COST_MAE={float(cm['MAE']):.6f}")
    print(f"{exp_id.upper()}_SCIENTIFIC_VERDICT={verdict}")
    return result
