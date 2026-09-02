"""Experiment D: future Cost-revision hazard as an auxiliary Delay signal.

Historical project prefixes are labelled for whether a material revised-cost
change occurs within 3/6/12 months.  Auxiliary classifiers are generated strictly
forward OOF; the final Delay residual model sees probabilities only, never the
future revision labels themselves.  Holdout project histories are not used to
fit the auxiliary models.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from backend.app.ml.experiments.post_exp113_delay_common import (
    fit_residual,
    numeric_design,
    persist,
    prepare_context,
    production_oof,
)

EXPERIMENT_ID = "exp_d"
EXPERIMENT_NAME = "D — future Cost-revision hazard for Delay"
SEED = 13301
HORIZONS = {"3m": 93, "6m": 186, "12m": 366}
AUX_INPUTS = [
    "cost_escalation_percentage",
    "expenditure_ratio",
    "progress_deviation",
    "schedule_slippage_days",
    "duration_ratio",
    "physical_progress",
    "approved_cost_cr",
    "exp34_cost_revision_count",
    "exp58_group_support",
]
HAZARD_FEATURES = [
    "exp_d_cost_revision_prob_3m",
    "exp_d_cost_revision_prob_6m",
    "exp_d_cost_revision_prob_12m",
    "exp_d_cost_revision_hazard_entropy",
    "exp_d_cost_revision_long_minus_short",
]
RESIDUAL_FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "expenditure_ratio",
    "cost_escalation_percentage",
    "exp58_delay_hier_prior",
    "exp58_group_support",
    *HAZARD_FEATURES,
]


def revision_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Build historical future-revision labels; never call this on the holdout."""
    out = frame.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["_order"] = np.arange(len(out))
    out = out.sort_values(["canonical_project_id", "snapshot_date", "_order"], kind="mergesort").copy()
    for name in HORIZONS:
        out[f"_label_{name}"] = 0

    for _, positions in out.groupby("canonical_project_id", sort=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        dates = out.iloc[pos]["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        costs = pd.to_numeric(out.iloc[pos].get("revised_cost_cr"), errors="coerce").to_numpy(float)
        if len(pos) < 2:
            continue
        event = np.zeros(len(pos), dtype=bool)
        valid = np.isfinite(costs[1:]) & np.isfinite(costs[:-1])
        event[1:] = valid & (np.abs(costs[1:] - costs[:-1]) > 1e-9)
        event_idx = np.flatnonzero(event)
        for local in range(len(pos)):
            future = event_idx[event_idx > local]
            if not len(future) or pd.isna(dates[local]):
                continue
            days = float((dates[future[0]] - dates[local]) / np.timedelta64(1, "D"))
            for name, horizon in HORIZONS.items():
                out.iloc[pos[local], out.columns.get_loc(f"_label_{name}")] = int(0 <= days <= horizon)
    return out.sort_values("_order", kind="mergesort").drop(columns=["_order"])


def _fit_aux(reference: pd.DataFrame, score: pd.DataFrame, seed: int) -> pd.DataFrame:
    labelled = revision_labels(reference)
    result = score.copy()
    probabilities = []
    for offset, (name, _days) in enumerate(HORIZONS.items()):
        target = pd.to_numeric(labelled[f"_label_{name}"], errors="coerce").fillna(0).astype(int)
        cols, _, x_train, x_score = numeric_design(labelled, result, AUX_INPUTS)
        if target.nunique() < 2 or not cols:
            prob = np.full(len(result), float(target.mean()) if len(target) else 0.0)
        else:
            model = LGBMClassifier(
                n_estimators=120,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=8,
                min_child_samples=60,
                reg_alpha=5,
                reg_lambda=25,
                random_state=seed + offset,
                verbosity=-1,
                n_jobs=1,
            )
            weight = pd.to_numeric(labelled["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
            model.fit(x_train, target.to_numpy(int), sample_weight=weight)
            prob = np.asarray(model.predict_proba(x_score)[:, 1], dtype=float)
        result[f"exp_d_cost_revision_prob_{name}"] = np.clip(prob, 0.0, 1.0)
        probabilities.append(np.clip(prob, 1e-6, 1 - 1e-6))

    p = np.vstack(probabilities).T
    result["exp_d_cost_revision_hazard_entropy"] = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p), axis=1)
    result["exp_d_cost_revision_long_minus_short"] = result["exp_d_cost_revision_prob_12m"] - result["exp_d_cost_revision_prob_3m"]
    return result


def _forward_aux_oof(oof: pd.DataFrame) -> pd.DataFrame:
    year_col = pd.to_numeric(oof["oof_year"], errors="coerce")
    years = sorted(int(v) for v in year_col.dropna().unique())
    parts = []
    for year in years[1:]:
        reference = oof.loc[year_col < year].copy()
        validation = oof.loc[year_col == year].copy()
        if len(reference) < 100 or validation.empty:
            continue
        parts.append(_fit_aux(reference, validation, SEED + year))
    if not parts:
        raise ValueError("Experiment D has no forward auxiliary folds")
    return pd.concat(parts, ignore_index=True)


def fit_experiment(training_end: int, output: str):
    ctx = prepare_context(training_end)
    oof = production_oof(ctx, max_folds=6)
    meta = _forward_aux_oof(oof)
    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_delay"]
    score = _fit_aux(oof, score, SEED + training_end)
    correction, details = fit_residual(meta, score, RESIDUAL_FEATURES, SEED)
    details.update(
        {
            "changed_dimension": "future_cost_revision_hazard_auxiliary",
            "auxiliary_horizons_days": HORIZONS,
            "auxiliary_predictions_for_meta": "strictly forward OOF",
            "holdout_future_revisions_used_for_fit": False,
            "future_revision_labels_are_model_features": False,
            "hazard_features": HAZARD_FEATURES,
        }
    )
    return persist(EXPERIMENT_ID, EXPERIMENT_NAME, ctx, ctx["production_delay"] + correction, details, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    fit_experiment(a.end, a.output)


if __name__ == "__main__":
    main()
