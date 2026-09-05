"""Dynamic per-row trust in the already-promoted Exp105 correction."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from backend.app.ml.experiments.post_exp105_cost_common import numeric_design, persist, prepare_context, production_oof

EXP_ID = "exp_cost_dynamic_exp105_scale"
NAME = "Dynamic Exp105 Correction-Strength Router"
MULTIPLIERS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
FEATURES = [
    "production_base", "production_prediction", "exp105_correction",
    "cost_escalation_percentage", "duration_ratio", "expenditure_ratio",
    "schedule_slippage_days", "approved_cost_cr", "exp12_history_12m",
    "exp12_cost_worsening_streak", "exp12_cost_revision_magnitude_12m_pct",
    "exp12_spend_vs_expected_progress_gap",
]


def _labels(frame: pd.DataFrame) -> np.ndarray:
    actual = pd.to_numeric(frame["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    base = pd.to_numeric(frame["production_base"], errors="coerce").to_numpy(float)
    correction = pd.to_numeric(frame["exp105_correction"], errors="coerce").to_numpy(float)
    losses = np.column_stack([np.abs(actual - (base + beta * correction)) for beta in MULTIPLIERS])
    return np.argmin(losses, axis=1).astype(int)


def _crossfit(oof: pd.DataFrame) -> tuple[pd.Series, dict]:
    years = sorted(int(v) for v in pd.to_numeric(oof["oof_year"], errors="coerce").dropna().unique())
    beta = pd.Series(1.0, index=oof.index, dtype=float)
    used = []
    for year in years[1:]:
        fit = oof[pd.to_numeric(oof["oof_year"], errors="coerce") < year].copy()
        val = oof[pd.to_numeric(oof["oof_year"], errors="coerce") == year].copy()
        if len(fit) < 100 or val.empty:
            continue
        fit["choice"] = _labels(fit)
        if fit["choice"].nunique() < 2:
            continue
        _, _, xfit, xval = numeric_design(fit, val, FEATURES)
        model = LGBMClassifier(
            objective="multiclass", n_estimators=120, learning_rate=0.03,
            max_depth=2, num_leaves=4, min_child_samples=50, reg_alpha=5,
            reg_lambda=30, random_state=10550 + year, verbosity=-1, n_jobs=1,
        )
        model.fit(xfit, fit["choice"].to_numpy(int), sample_weight=fit["sample_weight"].to_numpy(float))
        proba = model.predict_proba(xval)
        classes = np.asarray(model.classes_, dtype=int)
        expected = np.zeros(len(val), dtype=float)
        for j, cls in enumerate(classes):
            expected += proba[:, j] * MULTIPLIERS[int(cls)]
        negligible = np.abs(pd.to_numeric(val["exp105_correction"], errors="coerce").to_numpy(float)) < 0.25
        expected[negligible] = 1.0
        beta.loc[val.index] = expected
        used.append(year)
    return beta, {"meta_years": used}


def run(output: str) -> dict:
    ctx = prepare_context()
    oof = production_oof(ctx)
    beta, details = _crossfit(oof)
    actual = pd.to_numeric(oof["actual_cost_overrun_percentage"], errors="coerce").to_numpy(float)
    base = pd.to_numeric(oof["production_base"], errors="coerce").to_numpy(float)
    correction = pd.to_numeric(oof["exp105_correction"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").to_numpy(float)
    cap = float(np.nanquantile(np.abs(correction), 0.95))
    candidate_oof = base + np.clip(beta.to_numpy(float) * correction, -cap, cap)
    details["meta_oof_mae"] = float(np.average(np.abs(actual - candidate_oof), weights=weight))
    details["selected_cap"] = cap
    details["mean_beta"] = float(np.average(beta.to_numpy(float), weights=weight))

    train = oof.copy()
    train["choice"] = _labels(train)
    score = ctx["cohort"].copy()
    score["production_base"] = ctx["production_base"]
    score["production_prediction"] = ctx["production_prediction"]
    score["exp105_correction"] = ctx["production_correction"]
    _, _, xfit, xscore = numeric_design(train, score, FEATURES)
    model = LGBMClassifier(
        objective="multiclass", n_estimators=140, learning_rate=0.03,
        max_depth=2, num_leaves=4, min_child_samples=50, reg_alpha=5,
        reg_lambda=30, random_state=10599, verbosity=-1, n_jobs=1,
    )
    model.fit(xfit, train["choice"].to_numpy(int), sample_weight=train["sample_weight"].to_numpy(float))
    proba = model.predict_proba(xscore)
    classes = np.asarray(model.classes_, dtype=int)
    beta_score = np.zeros(len(score), dtype=float)
    for j, cls in enumerate(classes):
        beta_score += proba[:, j] * MULTIPLIERS[int(cls)]
    negligible = np.abs(ctx["production_correction"]) < 0.25
    beta_score[negligible] = 1.0
    prediction = ctx["production_base"] + np.clip(beta_score * ctx["production_correction"], -cap, cap)
    details.update({"holdout_used_for_selection": False, "full_holdout_retained": True})
    return persist(EXP_ID, NAME, ctx, prediction, details, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test-output/exp-cost-dynamic-exp105-scale/result.json")
    args = parser.parse_args()
    run(args.output)
