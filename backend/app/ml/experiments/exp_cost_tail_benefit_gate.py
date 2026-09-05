"""Positive-tail benefit-gated residual specialist over the frozen Exp105 anchor."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from backend.app.ml.experiments.post_exp105_cost_common import (
    numeric_design, persist, prepare_context, production_oof,
)

EXP_ID = "exp_cost_tail_benefit_gate"
NAME = "Positive-Tail Benefit-Gated Residual Specialist"
FEATURES = [
    "production_prediction", "production_base", "exp105_correction",
    "cost_escalation_percentage", "expenditure_ratio", "schedule_slippage_days",
    "duration_ratio", "approved_cost_cr", "revised_cost_cr",
    "cumulative_expenditure_cr", "elapsed_duration_days", "planned_duration_days",
    "exp12_history_12m", "exp12_cost_velocity_12m",
    "exp12_cost_revision_magnitude_12m_pct", "exp12_cost_worsening_streak",
    "exp12_spend_vs_expected_progress_gap",
]
QUANTILES = (0.50, 0.60, 0.70)
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _specialist(fit: pd.DataFrame, score: pd.DataFrame, alpha: float, seed: int) -> np.ndarray:
    cols, _, xfit, xscore = numeric_design(fit, score, FEATURES)
    target = np.log1p(np.maximum(pd.to_numeric(fit["residual"], errors="coerce").fillna(0).to_numpy(float), 0.0))
    weight = pd.to_numeric(fit["sample_weight"], errors="coerce").fillna(0).to_numpy(float)
    model = LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=180, learning_rate=0.025,
        max_depth=3, num_leaves=8, min_child_samples=50, reg_alpha=5, reg_lambda=30,
        random_state=seed, verbosity=-1, n_jobs=1,
    )
    model.fit(xfit, target, sample_weight=weight)
    return np.maximum(np.expm1(np.asarray(model.predict(xscore), dtype=float)), 0.0)


def _crossfit_specialist(oof: pd.DataFrame, alpha: float) -> pd.Series:
    years = sorted(int(v) for v in pd.to_numeric(oof["oof_year"], errors="coerce").dropna().unique())
    pred = pd.Series(np.nan, index=oof.index, dtype=float)
    for year in years[1:]:
        fit = oof[pd.to_numeric(oof["oof_year"], errors="coerce") < year]
        val = oof[pd.to_numeric(oof["oof_year"], errors="coerce") == year]
        if len(fit) < 100 or val.empty:
            continue
        pred.loc[val.index] = _specialist(fit, val, alpha, 10500 + year)
    return pred


def _gate_predictions(oof: pd.DataFrame, specialist: pd.Series, alpha: float) -> tuple[pd.Series, dict]:
    work = oof.copy()
    work["specialist_correction"] = specialist
    work = work.dropna(subset=["specialist_correction"]).copy()
    actual = pd.to_numeric(work["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(work["production_prediction"], errors="coerce").to_numpy(float)
    candidate = anchor + work["specialist_correction"].to_numpy(float)
    work["benefit"] = (np.abs(actual - candidate) + 0.5 < np.abs(actual - anchor)).astype(int)
    years = sorted(int(v) for v in pd.to_numeric(work["oof_year"], errors="coerce").dropna().unique())
    gate = pd.Series(0.0, index=work.index, dtype=float)
    for year in years[1:]:
        fit = work[pd.to_numeric(work["oof_year"], errors="coerce") < year]
        val = work[pd.to_numeric(work["oof_year"], errors="coerce") == year]
        if len(fit) < 100 or val.empty or fit["benefit"].nunique() < 2:
            continue
        features = FEATURES + ["specialist_correction"]
        cols, _, xfit, xval = numeric_design(fit, val, features)
        model = LGBMClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=2, num_leaves=4,
            min_child_samples=50, reg_alpha=5, reg_lambda=30, class_weight="balanced",
            random_state=10600 + year, verbosity=-1, n_jobs=1,
        )
        model.fit(xfit, fit["benefit"].to_numpy(int), sample_weight=fit["sample_weight"].to_numpy(float))
        gate.loc[val.index] = model.predict_proba(xval)[:, 1]
    return gate, {"alpha": alpha, "benefit_rate": float(work["benefit"].mean()), "gate_rows": int(gate.notna().sum())}


def run(output: str) -> dict:
    ctx = prepare_context()
    oof = production_oof(ctx)
    actual = pd.to_numeric(oof["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(oof["production_prediction"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").to_numpy(float)
    best = None
    for alpha in QUANTILES:
        specialist = _crossfit_specialist(oof, alpha)
        gate, meta = _gate_predictions(oof, specialist, alpha)
        valid = specialist.notna() & gate.reindex(oof.index).notna()
        if int(valid.sum()) < 100:
            continue
        cap = float(np.nanquantile(np.maximum(pd.to_numeric(oof.loc[valid, "residual"], errors="coerce"), 0.0), 0.95))
        for scale in SCALES:
            correction = scale * gate.reindex(oof.index).fillna(0).to_numpy(float) * np.clip(specialist.fillna(0).to_numpy(float), 0, cap)
            pred = anchor + correction
            mae = float(np.average(np.abs(actual[valid] - pred[valid]), weights=weight[valid]))
            candidate = (mae, alpha, scale, cap, meta)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("No valid forward-OOF specialist/gate configuration")
    _, alpha, scale, cap, meta = best

    specialist_train = _specialist(oof, oof, alpha, 10701)
    train = oof.copy()
    train["specialist_correction"] = specialist_train
    y = pd.to_numeric(train["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    base = pd.to_numeric(train["production_prediction"], errors="coerce").to_numpy(float)
    train["benefit"] = (np.abs(y - (base + specialist_train)) + 0.5 < np.abs(y - base)).astype(int)

    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_prediction"]
    score["production_base"] = ctx["production_base"]
    score["exp105_correction"] = ctx["production_correction"]
    specialist_score = _specialist(oof, score, alpha, 10702)
    train_features = FEATURES + ["specialist_correction"]
    score["specialist_correction"] = specialist_score
    _, _, xfit, xscore = numeric_design(train, score, train_features)
    gate_model = LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=2, num_leaves=4,
        min_child_samples=50, reg_alpha=5, reg_lambda=30, class_weight="balanced",
        random_state=10703, verbosity=-1, n_jobs=1,
    )
    gate_model.fit(xfit, train["benefit"].to_numpy(int), sample_weight=train["sample_weight"].to_numpy(float))
    gate_score = gate_model.predict_proba(xscore)[:, 1]
    prediction = ctx["production_prediction"] + scale * gate_score * np.clip(specialist_score, 0, cap)
    details = {
        "selected_alpha": alpha, "selected_scale": scale, "correction_cap": cap,
        "meta_oof_mae": best[0], "gate": meta,
        "holdout_used_for_selection": False, "full_holdout_retained": True,
    }
    return persist(EXP_ID, NAME, ctx, prediction, details, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test-output/exp-cost-tail-benefit-gate/result.json")
    args = parser.parse_args()
    run(args.output)
