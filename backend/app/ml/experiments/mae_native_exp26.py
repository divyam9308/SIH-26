"""Experiment 26: MAE-native conditional-median regression.

This challenger isolates the training-loss hypothesis. It keeps the exact
current production-selected algorithm family, target-specific feature contract,
project-balanced weights, temporal split, seeds, and tree hyperparameters, and
changes only the regression loss/criterion from squared error to absolute
error (L1/MAE).

Evaluation deliberately follows the promoted Experiment 12 scientific contract:
models train on the normal training window, but production and challenger are
scored on the same Exp12 filtered comparable future cohort -- snapshots with at
least MIN_HISTORY official observations in the trailing 12 months. The filtered
cohort is reweighted project-balanced after filtering, exactly as Exp12 does.
The future holdout is never used for model-family, feature, or loss selection.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.trajectory_exp12 import MIN_HISTORY
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_26"
EXPERIMENT_NAME = "MAE-native conditional-median regression"
EXPERIMENT_SCOPE = "cost+delay"
EXPERIMENT_SEQUENCE = 26
DELAY_SEED = 26204
COMPARISON_FILTER = f">={MIN_HISTORY} official observations in trailing 12 months (Exp12 contract)"


def _mae_regressors(seed: int) -> dict[str, object]:
    """Mirror production regressors exactly except for the regression loss."""
    return {
        "lightgbm": LGBMRegressor(
            n_estimators=240,
            learning_rate=.035,
            max_depth=5,
            num_leaves=24,
            objective="regression_l1",
            random_state=seed,
            verbosity=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=240,
            learning_rate=.035,
            max_depth=4,
            subsample=.85,
            colsample_bytree=.85,
            objective="reg:absoluteerror",
            random_state=seed,
            n_jobs=2,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=260,
            min_samples_leaf=3,
            max_features=.8,
            criterion="absolute_error",
            random_state=seed,
            n_jobs=2,
        ),
    }


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _target_verdict(gain: float) -> str:
    if gain > 0:
        return "PROMOTION CANDIDATE"
    if gain == 0:
        return "NO CHANGE"
    return "REGRESSION / DO NOT PROMOTE"


def _overall_verdict(cost_gain: float, delay_gain: float) -> str:
    # PR-level verdict is conservative. Target verdicts remain separate so a
    # later promotion can be target-only, matching the Exp12 promotion policy.
    if cost_gain >= 0 and delay_gain >= 0 and (cost_gain > 0 or delay_gain > 0):
        return "PROMOTION CANDIDATE"
    return "REGRESSION / DO NOT PROMOTE"


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _exp12_comparable(test: pd.DataFrame) -> pd.DataFrame:
    if "exp12_history_12m" not in test.columns:
        raise ValueError("Experiment 26 requires Exp12 trajectory history for comparable-cohort filtering.")
    compare = test[
        pd.to_numeric(test.exp12_history_12m, errors="coerce")
        .fillna(0)
        .ge(MIN_HISTORY)
    ].copy()
    if compare.canonical_project_id.nunique() < 2:
        raise ValueError("Experiment 26 has too few future projects on the Exp12 comparable cohort.")
    return assign_project_balanced_weights(compare)


def fit_experiment(
    *,
    data,
    training_start,
    training_end,
    test_end,
    production_bundle,
    production_receipt,
    **_,
):
    """Fit same-family L1 challengers and score on the Exp12 comparable cohort."""
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched.snapshot_date, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    selected = dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    cost_name = selected.get("cost")
    delay_name = selected.get("delay")
    available = _mae_regressors(PRODUCTION_COST_SEED)
    if cost_name not in available or delay_name not in available:
        raise ValueError(
            "Experiment 26 requires production-selected cost/delay families from "
            f"{sorted(available)}; got cost={cost_name!r}, delay={delay_name!r}."
        )

    cost_features = list(contract["cost"])
    delay_features = list(contract["delay"])
    if not cost_features or not delay_features:
        raise ValueError("Experiment 26 requires explicit production target feature contracts.")
    missing_cost = [name for name in cost_features if name not in train.columns]
    missing_delay = [name for name in delay_features if name not in train.columns]
    if missing_cost or missing_delay:
        raise ValueError(
            f"Experiment 26 feature contract mismatch: cost={missing_cost}, delay={missing_delay}"
        )

    # Only the loss changes. The family and all other hyperparameters are held
    # to the production-selected family for that target/window.
    cost_model = _fit_pipeline(
        _mae_regressors(PRODUCTION_COST_SEED)[cost_name],
        train,
        cost_features,
        "actual_cost_overrun_percentage",
    )
    delay_model = _fit_pipeline(
        _mae_regressors(DELAY_SEED)[delay_name],
        train,
        delay_features,
        "actual_delay_days",
    )

    compare = _exp12_comparable(test)
    compare["production_cost"] = production_bundle["cost"].predict(compare[cost_features])
    compare["production_delay"] = np.maximum(
        0, production_bundle["delay"].predict(compare[delay_features])
    )
    compare["experiment_cost"] = cost_model.predict(compare[cost_features])
    compare["experiment_delay"] = np.maximum(0, delay_model.predict(compare[delay_features]))

    prod_cost = _regression_metrics(
        compare.actual_cost_overrun_percentage,
        compare.production_cost.to_numpy(),
        compare.sample_weight,
        compare.canonical_project_id,
    )
    exp_cost = _regression_metrics(
        compare.actual_cost_overrun_percentage,
        compare.experiment_cost.to_numpy(),
        compare.sample_weight,
        compare.canonical_project_id,
    )
    prod_delay = _regression_metrics(
        compare.actual_delay_days,
        compare.production_delay.to_numpy(),
        compare.sample_weight,
        compare.canonical_project_id,
    )
    exp_delay = _regression_metrics(
        compare.actual_delay_days,
        compare.experiment_delay.to_numpy(),
        compare.sample_weight,
        compare.canonical_project_id,
    )

    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    overall_verdict = _overall_verdict(cost_gain, delay_gain)
    paired_cost = paired_project_mae_comparison(
        compare,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost",
        candidate_prediction="experiment_cost",
    )
    paired_delay = paired_project_mae_comparison(
        compare,
        actual="actual_delay_days",
        baseline_prediction="production_delay",
        candidate_prediction="experiment_delay",
    )

    union_features = list(dict.fromkeys(cost_features + delay_features))
    lookup = {
        _key(row): {name: row.get(name) for name in union_features}
        for _, row in compare.iterrows()
    }
    comparable = set(lookup)
    run_id = f"exp26-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"

    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": run_id,
            "model_role": "experiment",
            "promotion_allowed": False,
            "hypothesis": "MAE evaluation should be better aligned with conditional-median/L1 training than squared-error training.",
            "controlled_change": "regression loss/criterion only",
            "comparison_filter": COMPARISON_FILTER,
            "selected_algorithms": selected,
            "losses": {
                "lightgbm": "regression_l1",
                "xgboost": "reg:absoluteerror",
                "extra_trees": "absolute_error",
            },
            "production_feature_contract": {
                "cost": cost_features,
                "delay": delay_features,
            },
            "metrics": {"cost": exp_cost, "delay": exp_delay},
            "target_verdicts": {
                "cost": _target_verdict(cost_gain),
                "delay": _target_verdict(delay_gain),
            },
            "decision": overall_verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(compare.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "comparison_filter": COMPARISON_FILTER,
            "comparison_weighting": "project-balanced after Exp12 cohort filtering",
            "paired_project_cost_comparison": paired_cost,
            "paired_project_delay_comparison": paired_delay,
            "cost_algorithm_family": cost_name,
            "delay_algorithm_family": delay_name,
            "cost_target_verdict": _target_verdict(cost_gain),
            "delay_target_verdict": _target_verdict(delay_gain),
            "controlled_change": "loss_only",
            "decision": overall_verdict,
        },
        "runtime_state": {
            "cost_model": cost_model,
            "delay_model": delay_model,
            "cost_features": cost_features,
            "delay_features": delay_features,
            "lookup": lookup,
            "comparable": comparable,
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("Snapshot is outside the Experiment 12 filtered comparable cohort.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    cost_x = candidate.to_frame().T.reindex(columns=state["cost_features"])
    delay_x = candidate.to_frame().T.reindex(columns=state["delay_features"])
    return {
        "predicted_cost_overrun": round(float(state["cost_model"].predict(cost_x)[0]), 4),
        "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(delay_x)[0])), 4),
        "training_loss": "MAE/L1",
        "comparison_cohort": "Exp12 filtered comparable cohort",
    }
