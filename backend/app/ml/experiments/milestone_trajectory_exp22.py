"""Experiment 22: milestone-achievement trajectory features.

Milestone values are parsed from the same official PAIMANA snapshot (for example
"6/14"). Trajectory features are engineered on the full official monthly history
and then joined onto the quarterly supervised frame. Every value at snapshot t
uses only t and earlier rows of that same canonical project; missing milestone
eras remain missing and no future/backfill imputation is performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import TRAJECTORIES, build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    enrich_supervised_for_production,
    target_feature_contract,
    train_window_with_promoted_cost,
)

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "reports" / "experiments" / "exp_22"
EXPERIMENT_ID = "exp_22"
EXPERIMENT_NAME = "Milestone achievement trajectory"
EXPERIMENT_SCOPE = "cost_delay"
MILESTONE_FEATURES = [
    "exp22_milestones_achieved", "exp22_milestones_total", "exp22_milestone_ratio",
    "exp22_milestones_remaining", "exp22_milestone_velocity", "exp22_milestone_delta",
    "exp22_milestone_stagnant", "exp22_months_since_milestone_change",
]


def add_milestone_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineer milestone features causally on whatever history frame is supplied."""
    result = frame.copy()
    status = result.get("milestone_status", pd.Series(None, index=result.index)).astype("string")
    parts = status.str.extract(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
    result["exp22_milestones_achieved"] = pd.to_numeric(parts["done"], errors="coerce")
    result["exp22_milestones_total"] = pd.to_numeric(parts["total"], errors="coerce")
    total = result["exp22_milestones_total"]
    done = result["exp22_milestones_achieved"]
    result["exp22_milestone_ratio"] = (done / total).where(total.gt(0)).clip(0, 1)
    result["exp22_milestones_remaining"] = (total - done).where(total.notna() & done.notna()).clip(lower=0)
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce")
    result["exp22_milestone_velocity"] = np.nan
    result["exp22_milestone_delta"] = np.nan
    result["exp22_milestone_stagnant"] = np.nan
    result["exp22_months_since_milestone_change"] = np.nan

    ordered = result.sort_values(["canonical_project_id", "snapshot_date"])
    for _, group in ordered.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group["snapshot_date"]
        ratios = group["exp22_milestone_ratio"]
        achieved = group["exp22_milestones_achieved"]
        months = dates.diff().dt.days / 30.4375
        delta = achieved.diff()
        velocity = ratios.diff().div(months).where(months.gt(0))
        stagnant = pd.Series(np.where(delta.notna(), (delta <= 0).astype(float), np.nan), index=idx)
        since = pd.Series(np.nan, index=idx, dtype=float)
        last_change = None
        previous = np.nan
        for row_index in idx:
            current = result.at[row_index, "exp22_milestones_achieved"]
            current_date = result.at[row_index, "snapshot_date"]
            if pd.isna(current) or pd.isna(current_date):
                continue
            if pd.isna(previous) or current != previous:
                last_change = current_date
            if last_change is not None:
                since.at[row_index] = max(0.0, (current_date - last_change).days / 30.4375)
            previous = current
        result.loc[idx, "exp22_milestone_velocity"] = velocity.to_numpy()
        result.loc[idx, "exp22_milestone_delta"] = delta.to_numpy()
        result.loc[idx, "exp22_milestone_stagnant"] = stagnant.to_numpy()
        result.loc[idx, "exp22_months_since_milestone_change"] = since.to_numpy()
    return result


def enrich_with_monthly_milestones(frame: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join features engineered on full monthly history to supervised snapshots."""
    supervised = frame.copy()
    supervised["snapshot_date"] = pd.to_datetime(supervised["snapshot_date"], errors="coerce")
    supervised["canonical_project_id"] = supervised["canonical_project_id"].astype("string")
    if history is None:
        if TRAJECTORIES.exists():
            history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
        else:
            history = supervised.copy()
    monthly = history.copy()
    monthly["snapshot_date"] = pd.to_datetime(monthly["snapshot_date"], errors="coerce")
    monthly["canonical_project_id"] = monthly["canonical_project_id"].astype("string")
    monthly = add_milestone_features(monthly)
    lookup = monthly[["canonical_project_id", "snapshot_date", *MILESTONE_FEATURES]].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )
    supervised = supervised.drop(columns=[name for name in MILESTONE_FEATURES if name in supervised], errors="ignore")
    return supervised.merge(lookup, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")


def _improvement(base: float, challenger: float) -> float:
    return (base - challenger) / base * 100.0 if base else 0.0


def _decision(cost_improvement: float, delay_improvement: float) -> str:
    return "PROMOTION CANDIDATE" if cost_improvement >= 0 and delay_improvement >= 0 and (cost_improvement > 0 or delay_improvement > 0) else "REGRESSION / DO NOT PROMOTE"


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    data, identity = build_training_dataset()
    temp_root = Path(tempfile.mkdtemp(prefix="sih-exp22-"))
    run_id = f"exp22-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    try:
        production = train_window_with_promoted_cost(training_start, training_end, test_end, data=data, identity=identity, artifact_root=temp_root)
        artifact_dir = temp_root / f"{training_start}_{training_end}"
        metadata = production["metadata"]
        contract = target_feature_contract(metadata)
        selected = metadata["selected_algorithms"]

        enriched = enrich_supervised_for_production(data.copy())
        enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
        enriched = enrich_with_monthly_milestones(enriched)
        train, test = temporal_project_split(enriched, training_start, training_end, test_end)

        prod_cost = joblib.load(artifact_dir / "cost_model.pkl")
        prod_delay = joblib.load(artifact_dir / "delay_model.pkl")
        base_cost_pred = prod_cost.predict(test[contract["cost"]])
        base_delay_pred = np.maximum(0, prod_delay.predict(test[contract["delay"]]))

        cost_features = list(dict.fromkeys(contract["cost"] + MILESTONE_FEATURES))
        delay_features = list(dict.fromkeys(contract["delay"] + MILESTONE_FEATURES))
        cost_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[selected["cost"]], train, cost_features, "actual_cost_overrun_percentage")
        delay_model = _fit_pipeline(_regressors(26204)[selected["delay"]], train, delay_features, "actual_delay_days")
        exp_cost_pred = cost_model.predict(test[cost_features])
        exp_delay_pred = np.maximum(0, delay_model.predict(test[delay_features]))

        base_cost = _regression_metrics(test.actual_cost_overrun_percentage, base_cost_pred, test.sample_weight, test.canonical_project_id)
        exp_cost = _regression_metrics(test.actual_cost_overrun_percentage, exp_cost_pred, test.sample_weight, test.canonical_project_id)
        base_delay = _regression_metrics(test.actual_delay_days, base_delay_pred, test.sample_weight, test.canonical_project_id)
        exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
        cost_gain = _improvement(float(base_cost["MAE"]), float(exp_cost["MAE"]))
        delay_gain = _improvement(float(base_delay["MAE"]), float(exp_delay["MAE"]))

        train_ratio = train["exp22_milestone_ratio"].notna()
        test_ratio = test["exp22_milestone_ratio"].notna()
        report = {
            "experiment": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "status": "complete", "decision": _decision(cost_gain, delay_gain),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_period": [training_start, training_end], "testing_period": [training_end + 1, test_end],
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "production_selected_algorithms": selected, "added_features": MILESTONE_FEATURES,
            "metrics": {
                "production_cost_mae": float(base_cost["MAE"]), "experiment_cost_mae": float(exp_cost["MAE"]), "cost_improvement_percentage": cost_gain,
                "production_delay_mae": float(base_delay["MAE"]), "experiment_delay_mae": float(exp_delay["MAE"]), "delay_improvement_percentage": delay_gain,
                "production_cost": base_cost, "experiment_cost": exp_cost, "production_delay": base_delay, "experiment_delay": exp_delay,
            },
            "coverage": {
                "feature_history_granularity": "full official monthly history",
                "training_milestone_snapshot_share": float(train_ratio.mean()),
                "test_milestone_snapshot_share": float(test_ratio.mean()),
                "training_projects_with_milestones": int(train.loc[train_ratio, "canonical_project_id"].nunique()),
                "test_projects_with_milestones": int(test.loc[test_ratio, "canonical_project_id"].nunique()),
                "test_projects": int(test.canonical_project_id.nunique()), "test_snapshots": int(len(test)),
            },
            "leakage_policy": "Milestone numerator/denominator comes from the current snapshot. Velocity, delta, stagnation and months-since-change are engineered on the full official monthly history using only prior/current rows of the same project, then joined to supervised snapshots; no future backfill is used.",
            "production_changed": False,
        }
        out = REPORT_ROOT / f"{training_start}_{training_end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        validation = test[["canonical_project_id", "project_name", "snapshot_date", "completion_year", "milestone_status", "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight"]].copy()
        validation["production_cost_prediction"] = base_cost_pred
        validation["experiment_cost_prediction"] = exp_cost_pred
        validation["production_delay_prediction"] = base_delay_pred
        validation["experiment_delay_prediction"] = exp_delay_pred
        validation.to_csv(out / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
