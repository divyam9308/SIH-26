"""Exp130: one leakage-safe outer residual correction above current production.

The current Exp105 Cost + Exp113 Delay stack stays frozen as the anchor.  Fresh
forward production OOF predictions are converted into residual targets, and one
CatBoost residual model per target learns only the remaining systematic error.
Correction scale and clipping are chosen from training-only OOF evidence.
"""
from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.experiments.post_exp113_delay_common import forward_folds
from backend.app.ml.monthly_lifecycle import TRAJECTORY_FEATURES, assign_project_balanced_weights, build_training_dataset
from backend.app.ml.monthly_training import _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp35_baseline import (
    CALIBRATION_GATE_FEATURE,
    _aft_routing_limit,
    _select_aft_calibration_projects,
)
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_exp105_exp113_baseline import (
    PRODUCTION_COST_BASELINE,
    PRODUCTION_DELAY_BASELINE,
    train_window_with_promoted_cost_and_delay as train_current_production,
)

EXPERIMENT_ID = "exp130"
EXPERIMENT_NAME = "Outer production residual correction"
TRAINING_START = 2001
TRAINING_END = 2021
TEST_START = 2022
TEST_END = 2025

TARGETS = {
    "cost": "actual_cost_overrun_percentage",
    "delay": "actual_delay_days",
}
CATEGORICAL_FEATURES = ["sector", "implementing_agency", "lifecycle_stage"]
NUMERIC_FEATURES = [
    "production_prediction",
    "elapsed_duration_days",
    "physical_progress",
    "expenditure_ratio",
    "planned_remaining_days",
    "revised_remaining_days",
    "approved_cost_cr",
    "revised_cost_cr",
    "cost_intensity_per_day",
    "cost_escalation_percentage",
    "progress_deviation",
    "schedule_slippage_days",
    "duration_ratio",
    "prediction_confidence_proxy",
    "lifecycle_fraction",
    "production_x_lifecycle_stage",
    *TRAJECTORY_FEATURES,
]
RESIDUAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FORBIDDEN_MODEL_FEATURES = {
    "actual_cost_overrun_percentage",
    "actual_delay_days",
    "actual_risk",
    "completion_date",
    "actual_completion_date",
    "reported_completion_expenditure_cr",
    "residual",
    "cost_residual",
    "delay_residual",
}

_OOF_DATA = None
_OOF_IDENTITY = None


def metric(frame: pd.DataFrame, actual: str, prediction: np.ndarray) -> float:
    result = _regression_metrics(
        frame[actual],
        np.asarray(prediction, dtype=float),
        frame["sample_weight"],
        frame["canonical_project_id"],
    )
    return float(result["MAE"])


def gain(base: float, candidate: float) -> float:
    return (float(base) - float(candidate)) / float(base) * 100.0 if float(base) else 0.0


def _series(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def add_residual_features(frame: pd.DataFrame, production_prediction: np.ndarray) -> pd.DataFrame:
    """Build as-of residual-model features without consulting any outcome/error."""
    work = frame.copy()
    work["production_prediction"] = np.asarray(production_prediction, dtype=float)

    snapshot = pd.to_datetime(_series(work, "snapshot_date"), errors="coerce")
    planned = pd.to_datetime(_series(work, "planned_completion_date"), errors="coerce")
    revised = pd.to_datetime(_series(work, "revised_completion_date"), errors="coerce")
    work["planned_remaining_days"] = (planned - snapshot).dt.days.astype(float)
    work["revised_remaining_days"] = (revised - snapshot).dt.days.astype(float)

    approved = pd.to_numeric(_series(work, "approved_cost_cr"), errors="coerce")
    revised_cost = pd.to_numeric(_series(work, "revised_cost_cr"), errors="coerce")
    duration = pd.to_numeric(_series(work, "planned_duration_days"), errors="coerce").clip(lower=1)
    cost_basis = revised_cost.where(revised_cost.gt(0), approved)
    work["cost_intensity_per_day"] = cost_basis / duration

    duration_ratio = pd.to_numeric(_series(work, "duration_ratio"), errors="coerce")
    physical = pd.to_numeric(_series(work, "physical_progress"), errors="coerce") / 100.0
    duration_fraction = duration_ratio.clip(lower=0.0, upper=1.5)
    physical_fraction = physical.clip(lower=0.0, upper=1.0)
    lifecycle = duration_fraction.copy()
    both = duration_fraction.notna() & physical_fraction.notna()
    lifecycle.loc[both] = 0.5 * duration_fraction.loc[both] + 0.5 * physical_fraction.loc[both]
    lifecycle = lifecycle.fillna(physical_fraction).fillna(0.0).clip(lower=0.0, upper=1.5)
    work["lifecycle_fraction"] = lifecycle
    work["lifecycle_stage"] = pd.cut(
        lifecycle,
        bins=[-np.inf, 0.33, 0.67, 1.0, np.inf],
        labels=["EARLY", "MID", "LATE", "OVERDUE"],
        right=False,
    ).astype("string").fillna("UNKNOWN")
    work["production_x_lifecycle_stage"] = work["production_prediction"] * lifecycle

    identity = pd.to_numeric(_series(work, "identity_confidence", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    support = pd.to_numeric(_series(work, "exp58_group_support", 0.0), errors="coerce").fillna(0.0).clip(lower=0)
    support_conf = 1.0 - np.exp(-support / 20.0)
    completeness_cols = [
        "physical_progress",
        "expenditure_ratio",
        "revised_cost_cr",
        "schedule_slippage_days",
        *TRAJECTORY_FEATURES,
    ]
    available = pd.DataFrame(
        {c: _series(work, c).notna().astype(float) for c in completeness_cols},
        index=work.index,
    ).mean(axis=1)
    work["prediction_confidence_proxy"] = (identity + support_conf + available) / 3.0

    for col in CATEGORICAL_FEATURES:
        work[col] = _series(work, col, "UNKNOWN").astype("string").fillna("UNKNOWN")
    return work


def _design(train: pd.DataFrame, score: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    left = pd.DataFrame(index=train.index)
    right = pd.DataFrame(index=score.index)
    medians: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        a = pd.to_numeric(_series(train, col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(_series(score, col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(a.median()) if a.notna().any() else 0.0
        medians[col] = median
        left[col] = a.fillna(median)
        right[col] = b.fillna(median)
    for col in CATEGORICAL_FEATURES:
        left[col] = _series(train, col, "UNKNOWN").astype("string").fillna("UNKNOWN").astype(str)
        right[col] = _series(score, col, "UNKNOWN").astype("string").fillna("UNKNOWN").astype(str)
    return left[RESIDUAL_FEATURES], right[RESIDUAL_FEATURES], medians


def _model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=240,
        depth=5,
        learning_rate=0.03,
        loss_function="MAE",
        l2_leaf_reg=12.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _weighted_mae(actual, prediction, weight) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(prediction) & np.isfinite(weight) & (weight >= 0)
    if not mask.any():
        return float("inf")
    w = weight[mask]
    err = np.abs(actual[mask] - prediction[mask])
    return float(np.average(err, weights=w)) if float(w.sum()) > 0 else float(np.mean(err))


def fit_residual_model(
    oof: pd.DataFrame,
    score: pd.DataFrame,
    *,
    target: str,
    seed: int,
    nonnegative_final: bool,
) -> tuple[np.ndarray, dict]:
    """Nested-forward scale selection, then one final residual model."""
    if FORBIDDEN_MODEL_FEATURES.intersection(RESIDUAL_FEATURES):
        raise AssertionError("Target/error leakage entered the residual feature contract")

    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(x) for x in year_col.dropna().unique())
    meta = []
    for year in years[1:]:
        fit = oof.loc[year_col < year].copy()
        val = oof.loc[year_col == year].copy()
        if fit["canonical_project_id"].nunique() < 20 or val["canonical_project_id"].nunique() < 3:
            continue
        x_fit, x_val, _ = _design(fit, val)
        model = _model(seed + year)
        residual = pd.to_numeric(fit["residual"], errors="coerce").fillna(0.0).to_numpy(float)
        weight = pd.to_numeric(fit["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
        model.fit(
            x_fit,
            residual,
            sample_weight=weight,
            cat_features=CATEGORICAL_FEATURES,
        )
        cap = max(float(np.nanquantile(np.abs(residual), 0.90)), 1e-9)
        correction = np.clip(np.asarray(model.predict(x_val), dtype=float), -cap, cap)
        meta.append((val, correction))

    if not meta:
        raise ValueError(f"No nested-forward residual evidence for {target}")

    best = (float("inf"), 0.0)
    scale_scores = {}
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
        fold_mae = []
        fold_weight = []
        for val, correction in meta:
            actual = pd.to_numeric(val[target], errors="coerce").to_numpy(float)
            anchor = pd.to_numeric(val["production_prediction"], errors="coerce").to_numpy(float)
            prediction = anchor + scale * correction
            if nonnegative_final:
                prediction = np.maximum(0.0, prediction)
            weight = pd.to_numeric(val["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
            fold_mae.append(_weighted_mae(actual, prediction, weight))
            fold_weight.append(max(float(weight.sum()), 1e-9))
        score_mae = float(np.average(fold_mae, weights=fold_weight))
        scale_scores[str(scale)] = score_mae
        candidate = (score_mae, scale)
        if candidate < best:
            best = candidate

    selected_scale = float(best[1])
    x_oof, x_score, medians = _design(oof, score)
    final_model = _model(seed)
    residual = pd.to_numeric(oof["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    final_model.fit(
        x_oof,
        residual,
        sample_weight=weight,
        cat_features=CATEGORICAL_FEATURES,
    )
    cap = max(float(np.nanquantile(np.abs(residual), 0.90)), 1e-9)
    correction = selected_scale * np.clip(
        np.asarray(final_model.predict(x_score), dtype=float),
        -cap,
        cap,
    )
    details = {
        "model": "CatBoostRegressor",
        "selected_scale": selected_scale,
        "scale_scores": scale_scores,
        "correction_cap_q90_abs_oof_residual": cap,
        "features": RESIDUAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_medians": medians,
        "meta_oof_years": [int(v["oof_year"].iloc[0]) for v, _ in meta],
        "holdout_used_for_fit_or_selection": False,
        "recursive_residual_stacking": False,
    }
    return correction, details


def _prepare_final_context() -> dict:
    data, identity = build_training_dataset()
    with tempfile.TemporaryDirectory(prefix="exp130-production-") as td:
        root = Path(td) / "models"
        train_current_production(
            TRAINING_START,
            TRAINING_END,
            TEST_END,
            data=data,
            identity=identity,
            artifact_root=root,
        )
        target = root / f"{TRAINING_START}_{TRAINING_END}"
        cost_model = joblib.load(target / "cost_model.pkl")
        delay_model = joblib.load(target / "delay_model.pkl")

        prepared = normalize_taxonomy(_prepare(data))
        train, test = temporal_project_split(prepared, TRAINING_START, TRAINING_END, TEST_END)
        train, test, _ = _build_temporal_delay_priors(train, test)
        cohort = _production_cost_evaluation_rows(test).copy()
        ids = _select_aft_calibration_projects(
            cohort,
            limit=_aft_routing_limit(TRAINING_START, TRAINING_END, TEST_END),
        )
        cohort[CALIBRATION_GATE_FEATURE] = (
            cohort["canonical_project_id"].astype("string").isin(ids)
        )
        cohort = assign_project_balanced_weights(cohort)
        production_cost = np.asarray(cost_model.predict(cohort), dtype=float)
        production_delay = np.maximum(0.0, np.asarray(delay_model.predict(cohort), dtype=float))

    return {
        "data": data,
        "identity": identity,
        "train": train,
        "cohort": cohort,
        "production_cost": production_cost,
        "production_delay": production_delay,
    }


def _init_oof_worker(data, identity):
    global _OOF_DATA, _OOF_IDENTITY
    _OOF_DATA = data
    _OOF_IDENTITY = identity


def _production_oof_fold(val: pd.DataFrame, year: int, data=None, identity=None) -> pd.DataFrame:
    source_data = _OOF_DATA if data is None else data
    source_identity = _OOF_IDENTITY if identity is None else identity
    if source_data is None or source_identity is None:
        raise RuntimeError("Exp130 OOF worker is not initialized")

    train_end = int(year) - 1
    with tempfile.TemporaryDirectory(prefix=f"exp130-oof-{year}-") as td:
        root = Path(td) / "models"
        train_current_production(
            TRAINING_START,
            train_end,
            int(year),
            data=source_data,
            identity=source_identity,
            artifact_root=root,
        )
        target = root / f"{TRAINING_START}_{train_end}"
        cost_model = joblib.load(target / "cost_model.pkl")
        delay_model = joblib.load(target / "delay_model.pkl")

        comparable = _production_cost_evaluation_rows(val).copy()
        ids = _select_aft_calibration_projects(
            comparable,
            limit=_aft_routing_limit(TRAINING_START, train_end, int(year)),
        )
        comparable[CALIBRATION_GATE_FEATURE] = (
            comparable["canonical_project_id"].astype("string").isin(ids)
        )
        comparable = assign_project_balanced_weights(comparable)
        cost_prediction = np.asarray(cost_model.predict(comparable), dtype=float)
        delay_prediction = np.maximum(0.0, np.asarray(delay_model.predict(comparable), dtype=float))

    cost = add_residual_features(comparable, cost_prediction)
    cost["residual"] = (
        pd.to_numeric(cost["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
        - cost_prediction
    )
    cost["target_kind"] = "cost"
    cost["oof_year"] = int(year)

    delay = add_residual_features(comparable, delay_prediction)
    delay["residual"] = (
        pd.to_numeric(delay["actual_delay_days"], errors="coerce").to_numpy(float)
        - delay_prediction
    )
    delay["target_kind"] = "delay"
    delay["oof_year"] = int(year)

    return pd.concat([cost, delay], ignore_index=True)


def production_oof(ctx: dict, max_folds: int = 5) -> pd.DataFrame:
    folds = [
        (val, int(year))
        for _, val, year in forward_folds(ctx["train"], max_folds=max_folds)
        if int(year) - 1 >= 2005
    ]
    requested = int(os.environ.get("EXP130_OOF_WORKERS", "1"))
    workers = max(1, min(requested, len(folds), 4))
    parts = []
    errors = []

    if workers == 1:
        for val, year in folds:
            try:
                parts.append(_production_oof_fold(val, year, ctx["data"], ctx["identity"]))
                print(f"EXP130_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
            except Exception as exc:
                errors.append(f"{year}: {type(exc).__name__}: {exc}")
                print(f"EXP130_PRODUCTION_OOF_FOLD_FAILED={errors[-1]}", flush=True)
    else:
        threads = max(1, min(int(os.environ.get("EXP130_THREADS_PER_WORKER", "2")), 4))
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(threads)
        os.environ["LOKY_MAX_CPU_COUNT"] = str(threads)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_oof_worker,
            initargs=(ctx["data"], ctx["identity"]),
        ) as pool:
            pending = {pool.submit(_production_oof_fold, val, year): year for val, year in folds}
            for future in as_completed(pending):
                year = pending[future]
                try:
                    parts.append(future.result())
                    print(f"EXP130_PRODUCTION_OOF_FOLD_COMPLETED={year}", flush=True)
                except Exception as exc:
                    errors.append(f"{year}: {type(exc).__name__}: {exc}")
                    print(f"EXP130_PRODUCTION_OOF_FOLD_FAILED={errors[-1]}", flush=True)

    if len(parts) < 3:
        detail = "; ".join(errors) if errors else "no eligible folds"
        raise ValueError(f"Need >=3 strict forward production OOF folds; completed={len(parts)}; {detail}")
    parts.sort(key=lambda frame: int(frame["oof_year"].iloc[0]))
    return pd.concat(parts, ignore_index=True)


def run(output: str | Path = "reports/exp130_outer_residual_2001_2021.json") -> dict:
    ctx = _prepare_final_context()
    oof = production_oof(ctx, max_folds=5)
    cohort = ctx["cohort"]

    cost_oof = oof.loc[oof["target_kind"].eq("cost")].copy()
    delay_oof = oof.loc[oof["target_kind"].eq("delay")].copy()

    cost_score = add_residual_features(cohort, ctx["production_cost"])
    delay_score = add_residual_features(cohort, ctx["production_delay"])

    cost_correction, cost_details = fit_residual_model(
        cost_oof,
        cost_score,
        target=TARGETS["cost"],
        seed=13001,
        nonnegative_final=False,
    )
    delay_correction, delay_details = fit_residual_model(
        delay_oof,
        delay_score,
        target=TARGETS["delay"],
        seed=13002,
        nonnegative_final=True,
    )

    experiment_cost = np.asarray(ctx["production_cost"], dtype=float) + cost_correction
    experiment_delay = np.maximum(
        0.0,
        np.asarray(ctx["production_delay"], dtype=float) + delay_correction,
    )

    production_cost_mae = metric(cohort, TARGETS["cost"], ctx["production_cost"])
    experiment_cost_mae = metric(cohort, TARGETS["cost"], experiment_cost)
    production_delay_mae = metric(cohort, TARGETS["delay"], ctx["production_delay"])
    experiment_delay_mae = metric(cohort, TARGETS["delay"], experiment_delay)

    cost_gain = gain(production_cost_mae, experiment_cost_mae)
    delay_gain = gain(production_delay_mae, experiment_delay_mae)
    verdict = (
        "PROMOTION CANDIDATE"
        if cost_gain > 0 and delay_gain > 0
        else "MIXED / DO NOT PROMOTE"
        if cost_gain > 0 or delay_gain > 0
        else "DO NOT PROMOTE"
    )

    result = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "scope": "cost+delay",
        "production_main_base": "PR #186 production stack",
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "production_delay_baseline": PRODUCTION_DELAY_BASELINE,
        "training_start": TRAINING_START,
        "training_end": TRAINING_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "comparison_test_projects": int(cohort["canonical_project_id"].nunique()),
        "comparison_test_snapshots": int(len(cohort)),
        "production_cost_mae": production_cost_mae,
        "experiment_cost_mae": experiment_cost_mae,
        "cost_improvement_percentage": round(cost_gain, 6),
        "production_delay_mae": production_delay_mae,
        "experiment_delay_mae": experiment_delay_mae,
        "delay_improvement_percentage": round(delay_gain, 6),
        "oof_years": sorted(int(x) for x in pd.to_numeric(oof["oof_year"]).dropna().unique()),
        "oof_projects_cost": int(cost_oof["canonical_project_id"].nunique()),
        "oof_projects_delay": int(delay_oof["canonical_project_id"].nunique()),
        "holdout_used_for_fit_or_selection": False,
        "recursive_residual_stacking": False,
        "correction_layer_count": 1,
        "cost_details": cost_details,
        "delay_details": delay_details,
        "execution_verdict": "EXECUTION VALID",
        "scientific_verdict": verdict,
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")

    print(f"EXP130_PRODUCTION_COST_MAE={production_cost_mae:.6f}")
    print(f"EXP130_EXPERIMENT_COST_MAE={experiment_cost_mae:.6f}")
    print(f"EXP130_COST_IMPROVEMENT_PERCENT={cost_gain:.6f}")
    print(f"EXP130_PRODUCTION_DELAY_MAE={production_delay_mae:.6f}")
    print(f"EXP130_EXPERIMENT_DELAY_MAE={experiment_delay_mae:.6f}")
    print(f"EXP130_DELAY_IMPROVEMENT_PERCENT={delay_gain:.6f}")
    print(f"EXP130_SCIENTIFIC_VERDICT={verdict}")
    return result
