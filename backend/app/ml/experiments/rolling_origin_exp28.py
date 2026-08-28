"""Experiment 28: rolling-origin robust model selection.

Current production chooses a regressor from one final validation year. This
challenger keeps the same candidate model families, hyperparameters, feature
contracts, targets and weights, but chooses the family using several historical
forward-origin validation years. The untouched future holdout is never consulted.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_28"
EXPERIMENT_NAME = "Rolling-origin robust model selection"
EXPERIMENT_SCOPE = "cost+delay"
EXPERIMENT_SEQUENCE = 28
DELAY_SEED = 26204
MAX_FOLDS = 4


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _rolling_folds(train: pd.DataFrame, max_folds: int = MAX_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    years = sorted(int(y) for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique())
    folds = []
    for year in reversed(years[1:]):
        fitting = train[pd.to_numeric(train.completion_year, errors="coerce").lt(year)].copy()
        validation = train[pd.to_numeric(train.completion_year, errors="coerce").eq(year)].copy()
        if fitting.canonical_project_id.nunique() >= 10 and validation.canonical_project_id.nunique() >= 3:
            folds.append((fitting, validation, year))
        if len(folds) >= max_folds:
            break
    return list(reversed(folds))


def _robust_select(train: pd.DataFrame, features: list[str], target: str, seed: int) -> tuple[str, list[dict]]:
    folds = _rolling_folds(train)
    if len(folds) < 2:
        raise ValueError("Experiment 28 requires at least two valid rolling-origin folds.")
    rows = []
    for name in _regressors(seed):
        fold_metrics = []
        for fitting, validation, year in folds:
            model = _fit_pipeline(_regressors(seed)[name], fitting, features, target)
            pred = model.predict(validation[features])
            if target == "actual_delay_days":
                pred = np.maximum(0, pred)
            metrics = _regression_metrics(
                validation[target], pred, validation.sample_weight, validation.canonical_project_id
            )
            fold_metrics.append({"year": year, "MAE": float(metrics["MAE"]), "projects": metrics["unique_projects"]})
        maes = [item["MAE"] for item in fold_metrics]
        rows.append({
            "algorithm": name,
            "mean_MAE": float(np.mean(maes)),
            "worst_MAE": float(np.max(maes)),
            "std_MAE": float(np.std(maes)),
            "folds": fold_metrics,
        })
    winner = min(rows, key=lambda item: (item["mean_MAE"], item["worst_MAE"], item["std_MAE"]))["algorithm"]
    for item in rows:
        item["selected"] = item["algorithm"] == winner
    return winner, rows


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched.snapshot_date, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(contract["delay"])

    cost_name, cost_selection = _robust_select(
        train, cost_features, "actual_cost_overrun_percentage", PRODUCTION_COST_SEED
    )
    delay_name, delay_selection = _robust_select(
        train, delay_features, "actual_delay_days", DELAY_SEED
    )
    cost_model = _fit_pipeline(
        _regressors(PRODUCTION_COST_SEED)[cost_name],
        train, cost_features, "actual_cost_overrun_percentage"
    )
    delay_model = _fit_pipeline(
        _regressors(DELAY_SEED)[delay_name],
        train, delay_features, "actual_delay_days"
    )

    cost_compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(cost_compare[cost_features])
    exp_cost_pred = cost_model.predict(cost_compare[cost_features])
    prod_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage, prod_cost_pred,
        cost_compare.sample_weight, cost_compare.canonical_project_id
    )
    exp_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage, exp_cost_pred,
        cost_compare.sample_weight, cost_compare.canonical_project_id
    )

    prod_delay_pred = np.maximum(0, production_bundle["delay"].predict(test[delay_features]))
    exp_delay_pred = np.maximum(0, delay_model.predict(test[delay_features]))
    prod_delay = _regression_metrics(test.actual_delay_days, prod_delay_pred, test.sample_weight, test.canonical_project_id)
    exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)

    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = (
        "PROMOTION CANDIDATE"
        if cost_gain >= 0 and delay_gain >= 0 and (cost_gain > 0 or delay_gain > 0)
        else "REGRESSION / DO NOT PROMOTE"
    )
    lookup = {
        _key(row): {name: row.get(name) for name in dict.fromkeys(cost_features + delay_features)}
        for _, row in test.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": f"exp28-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment",
            "promotion_allowed": False,
            "changed_dimension": "training_only_model_selection_policy",
            "selected_algorithms": {"cost": cost_name, "delay": delay_name},
            "rolling_origin_selection": {"cost": cost_selection, "delay": delay_selection},
            "future_holdout_used_for_selection": False,
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(test.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(test)),
            "cost_comparison_projects": int(cost_compare.canonical_project_id.nunique()),
            "cost_comparison_snapshots": int(len(cost_compare)),
            "selected_cost_algorithm": cost_name,
            "selected_delay_algorithm": delay_name,
            "selection_folds": len(_rolling_folds(train)),
            "decision": verdict,
        },
        "runtime_state": {
            "cost_model": cost_model,
            "delay_model": delay_model,
            "cost_features": cost_features,
            "delay_features": delay_features,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 28 feature vector is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    cost = float(state["cost_model"].predict(candidate.to_frame().T.reindex(columns=state["cost_features"]))[0])
    delay = max(0.0, float(state["delay_model"].predict(candidate.to_frame().T.reindex(columns=state["delay_features"]))[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
