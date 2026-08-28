"""Experiment 29: dense full-lifecycle path-dependence features.

Exp12 captures recent 3/6/12-month dynamics. Exp29 adds cumulative as-of path
summaries computed from the full official monthly history: revision counts,
historical peaks/recovery, persistence and worsening shares. No future snapshot
or completion outcome is used in feature construction.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import TRAJECTORIES
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_29"
EXPERIMENT_NAME = "Full-lifecycle path-dependence features"
EXPERIMENT_SCOPE = "cost+delay"
EXPERIMENT_SEQUENCE = 29
DELAY_SEED = 26204

PATH_FEATURES = [
    "exp29_observations_seen", "exp29_months_observed", "exp29_cost_revision_count",
    "exp29_schedule_revision_count", "exp29_cumulative_abs_cost_revision_pct",
    "exp29_max_cost_escalation", "exp29_cost_recovery_from_peak",
    "exp29_max_schedule_slippage", "exp29_delay_recovery_from_peak",
    "exp29_slippage_positive_share", "exp29_cost_overrun_positive_share",
    "exp29_cost_worsening_share", "exp29_delay_worsening_share",
    "exp29_months_since_first_cost_revision", "exp29_months_since_first_schedule_revision",
]


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _expanding_mean(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups, sort=False).expanding().mean().reset_index(level=0, drop=True).sort_index()


def _path_history(history: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy()
    hist["snapshot_date"] = pd.to_datetime(hist.snapshot_date, errors="coerce")
    hist["canonical_project_id"] = hist.canonical_project_id.astype("string")
    hist = hist.dropna(subset=["canonical_project_id", "snapshot_date"]).sort_values(["canonical_project_id", "snapshot_date"])
    gid = hist["canonical_project_id"]
    for col in ("revised_cost_cr", "cost_escalation_percentage", "schedule_slippage_days"):
        if col not in hist:
            hist[col] = np.nan
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist["exp29_observations_seen"] = hist.groupby("canonical_project_id").cumcount() + 1
    first_snapshot = hist.groupby("canonical_project_id").snapshot_date.transform("min")
    hist["exp29_months_observed"] = (hist.snapshot_date - first_snapshot).dt.days / 30.4375
    prev_cost = hist.groupby("canonical_project_id").revised_cost_cr.shift(1)
    cost_change = hist.revised_cost_cr.notna() & prev_cost.notna() & hist.revised_cost_cr.sub(prev_cost).abs().gt(1e-9)
    prev_slip = hist.groupby("canonical_project_id").schedule_slippage_days.shift(1)
    schedule_change = hist.schedule_slippage_days.notna() & prev_slip.notna() & hist.schedule_slippage_days.sub(prev_slip).abs().gt(1e-9)
    hist["exp29_cost_revision_count"] = cost_change.astype(int).groupby(gid).cumsum()
    hist["exp29_schedule_revision_count"] = schedule_change.astype(int).groupby(gid).cumsum()
    pct_revision = hist.revised_cost_cr.sub(prev_cost).abs().div(prev_cost.abs().replace(0, np.nan)).mul(100).fillna(0)
    hist["exp29_cumulative_abs_cost_revision_pct"] = pct_revision.groupby(gid).cumsum()
    hist["exp29_max_cost_escalation"] = hist.groupby("canonical_project_id").cost_escalation_percentage.cummax()
    hist["exp29_cost_recovery_from_peak"] = hist.exp29_max_cost_escalation - hist.cost_escalation_percentage
    hist["exp29_max_schedule_slippage"] = hist.groupby("canonical_project_id").schedule_slippage_days.cummax()
    hist["exp29_delay_recovery_from_peak"] = hist.exp29_max_schedule_slippage - hist.schedule_slippage_days
    slip_positive = hist.schedule_slippage_days.gt(0).astype(float).where(hist.schedule_slippage_days.notna())
    cost_positive = hist.cost_escalation_percentage.gt(0).astype(float).where(hist.cost_escalation_percentage.notna())
    cost_worsening = hist.cost_escalation_percentage.diff().gt(0).astype(float)
    delay_worsening = hist.schedule_slippage_days.diff().gt(0).astype(float)
    group_change = gid.ne(gid.shift())
    cost_worsening = cost_worsening.mask(group_change)
    delay_worsening = delay_worsening.mask(group_change)
    hist["exp29_slippage_positive_share"] = _expanding_mean(slip_positive, gid)
    hist["exp29_cost_overrun_positive_share"] = _expanding_mean(cost_positive, gid)
    hist["exp29_cost_worsening_share"] = _expanding_mean(cost_worsening, gid)
    hist["exp29_delay_worsening_share"] = _expanding_mean(delay_worsening, gid)
    first_cost_revision = hist.snapshot_date.where(cost_change).groupby(gid).transform("min")
    first_schedule_revision = hist.snapshot_date.where(schedule_change).groupby(gid).transform("min")
    hist["exp29_months_since_first_cost_revision"] = ((hist.snapshot_date - first_cost_revision).dt.days / 30.4375).fillna(-1)
    hist["exp29_months_since_first_schedule_revision"] = ((hist.snapshot_date - first_schedule_revision).dt.days / 30.4375).fillna(-1)
    return hist[["canonical_project_id", "snapshot_date", *PATH_FEATURES]].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")


def enrich_path_dependence(supervised: pd.DataFrame) -> pd.DataFrame:
    history = pd.read_csv(TRAJECTORIES, low_memory=False) if TRAJECTORIES.exists() else supervised.copy()
    engineered = _path_history(history)
    result = supervised.copy()
    result["snapshot_date"] = pd.to_datetime(result.snapshot_date, errors="coerce")
    result["canonical_project_id"] = result.canonical_project_id.astype("string")
    return result.merge(engineered, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    selected = dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    cost_name, delay_name = selected.get("cost"), selected.get("delay")
    if cost_name not in _regressors(PRODUCTION_COST_SEED) or delay_name not in _regressors(DELAY_SEED):
        raise ValueError(f"Exp29 unsupported production families: {selected}")
    cost_features = list(dict.fromkeys(list(contract["cost"]) + PATH_FEATURES))
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))
    cost_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[cost_name], train, cost_features, "actual_cost_overrun_percentage")
    delay_model = _fit_pipeline(_regressors(DELAY_SEED)[delay_name], train, delay_features, "actual_delay_days")
    cost_compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(cost_compare[list(contract["cost"])])
    exp_cost_pred = cost_model.predict(cost_compare[cost_features])
    prod_cost = _regression_metrics(cost_compare.actual_cost_overrun_percentage, prod_cost_pred, cost_compare.sample_weight, cost_compare.canonical_project_id)
    exp_cost = _regression_metrics(cost_compare.actual_cost_overrun_percentage, exp_cost_pred, cost_compare.sample_weight, cost_compare.canonical_project_id)
    prod_delay_pred = np.maximum(0, production_bundle["delay"].predict(test[list(contract["delay"])]))
    exp_delay_pred = np.maximum(0, delay_model.predict(test[delay_features]))
    prod_delay = _regression_metrics(test.actual_delay_days, prod_delay_pred, test.sample_weight, test.canonical_project_id)
    exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if cost_gain >= 0 and delay_gain >= 0 and (cost_gain > 0 or delay_gain > 0) else "REGRESSION / DO NOT PROMOTE"
    lookup = {_key(row): {name: row.get(name) for name in PATH_FEATURES} for _, row in test.iterrows()}
    return {
        "experiment": {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE, "run_id": f"exp29-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}", "model_role": "experiment", "promotion_allowed": False, "changed_dimension": "dense_as_of_path_feature_set", "added_features": PATH_FEATURES, "selected_algorithms": selected, "history_source": "full official monthly trajectory table", "future_holdout_used_for_feature_construction": False, "decision": verdict},
        "overall_comparison": {"production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": exp_cost["MAE"], "cost_improvement_percentage": round(cost_gain, 4), "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"], "delay_improvement_percentage": round(delay_gain, 4), "comparison_test_projects": int(test.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(test)), "cost_comparison_projects": int(cost_compare.canonical_project_id.nunique()), "path_feature_nonmissing_share": float(test[PATH_FEATURES].notna().any(axis=1).mean()), "decision": verdict},
        "runtime_state": {"cost_model": cost_model, "delay_model": delay_model, "cost_features": cost_features, "delay_features": delay_features, "lookup": lookup, "comparable": set(lookup)},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 29 path history is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items(): candidate[name] = value
    cost = float(state["cost_model"].predict(candidate.to_frame().T.reindex(columns=state["cost_features"]))[0])
    delay = max(0.0, float(state["delay_model"].predict(candidate.to_frame().T.reindex(columns=state["delay_features"]))[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
