"""Experiment 62 / U1: nonlinear rolling-OOF residual booster on Exp61.

The strong Exp61 predictions remain the anchor. Separate heavily regularized
LightGBM models learn only bounded Cost and Delay residual corrections from
rolling out-of-fold training errors. Future holdout outcomes are never used.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import (
    cost_oof_frame,
    delay_oof_frame,
    production_comparison,
    run_cli,
    weighted_quantile,
)

EXPERIMENT_ID = "exp_62"
EXPERIMENT_SEQUENCE = 62
MARKER = "EXP62"
EXPERIMENT_NAME = "U1 nonlinear cross-fitted residual booster on Exp61"
EXPERIMENT_SCOPE = "cost+delay"
CHANGED_DIMENSION = "bounded_nonlinear_oof_residual_correction"

CANDIDATES = [
    "production_prediction",
    "cost_escalation_percentage",
    "schedule_slippage_days",
    "duration_ratio",
    "expenditure_ratio",
    "progress_deviation",
    "approved_cost_cr",
    "exp58_delay_hier_prior",
    "exp58_group_support",
]


def _matrix(train: pd.DataFrame, score: pd.DataFrame):
    cols = [c for c in CANDIDATES if c in train.columns and c in score.columns]
    if "production_prediction" not in cols:
        raise AssertionError("production prediction missing from residual booster design")
    medians = {c: float(pd.to_numeric(train[c], errors="coerce").median()) for c in cols}
    x_train = pd.DataFrame({c: pd.to_numeric(train[c], errors="coerce").fillna(medians[c]) for c in cols})
    x_score = pd.DataFrame({c: pd.to_numeric(score[c], errors="coerce").fillna(medians[c]) for c in cols})
    return cols, x_train, x_score


def _fit_booster(oof: pd.DataFrame, score: pd.DataFrame, seed: int):
    cols, x_train, x_score = _matrix(oof, score)
    y = pd.to_numeric(oof["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    w = pd.to_numeric(oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=12,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=5.0,
        reg_lambda=25.0,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x_train, y, sample_weight=w)
    correction = np.asarray(model.predict(x_score), dtype=float)
    cap = weighted_quantile(np.abs(y), w, 0.90)
    cap = max(float(cap), 1e-9)
    correction = np.clip(correction, -cap, cap)
    return correction, {"features": cols, "correction_cap_abs_residual_q90": cap, "oof_rows": int(len(oof))}


def fit_experiment(*, data, production_bundle, training_start, training_end, test_end, **kwargs):
    _, _, cohort, production_cost, production_delay = production_comparison(
        data, production_bundle, training_start, training_end, test_end
    )

    cost_oof = cost_oof_frame(data, production_bundle, training_start, training_end, test_end)
    cost_score = cohort.copy()
    cost_score["production_prediction"] = production_cost
    cost_correction, cost_details = _fit_booster(cost_oof, cost_score, 6201)
    experiment_cost = production_cost + cost_correction

    delay_oof = delay_oof_frame(data, production_bundle, training_start, training_end, test_end)
    delay_score = cohort.copy()
    delay_score["production_prediction"] = production_delay
    delay_correction, delay_details = _fit_booster(delay_oof, delay_score, 6202)
    experiment_delay = np.maximum(0.0, production_delay + delay_correction)

    return _persist(
        EXPERIMENT_ID,
        EXPERIMENT_NAME,
        EXPERIMENT_SCOPE,
        CHANGED_DIMENSION,
        cohort,
        production_cost,
        experiment_cost,
        production_delay,
        experiment_delay,
        {
            "baseline": "assumed Exp61 production from PR #96",
            "cost": cost_details,
            "delay": delay_details,
            "holdout_used_for_fit_or_selection": False,
            "base_prediction_replaced": False,
            "correction_only": True,
        },
    )


if __name__ == "__main__":
    run_cli(sys.modules[__name__])
