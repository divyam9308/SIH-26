"""Experiment 23: leakage-safe state/geographic execution residual priors.

The challenger leaves production models untouched and learns small, shrunk
state and state-sector corrections from training-project residuals only. Each
training project contributes one residual so dense monthly histories cannot
outvote sparse projects. Missing/unseen geography receives zero correction.
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

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract, train_window_with_promoted_cost

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "reports" / "experiments" / "exp_23"
EXPERIMENT_ID = "exp_23"
EXPERIMENT_NAME = "State + state-sector execution residual priors"
EXPERIMENT_SCOPE = "cost_delay"
STATE_K = 18.0
STATE_SECTOR_K = 28.0
STATE_WEIGHT = 0.55
STATE_SECTOR_WEIGHT = 0.45
MAX_COST_CORRECTION = 15.0
MAX_DELAY_CORRECTION = 240.0
MISSING = "__missing__"


def _geo_key(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip().str.upper()
    return out.where(out.ne(""), MISSING)


def _state_sector(state: pd.Series, sector: pd.Series) -> pd.Series:
    s = _geo_key(state)
    c = sector.fillna("").astype(str).str.strip().str.upper().replace("", MISSING)
    return s + "||" + c


def project_level_residuals(train: pd.DataFrame, predictions: np.ndarray, target: str) -> pd.DataFrame:
    rows = train[["canonical_project_id", "state", "sector", target]].copy()
    rows["prediction"] = np.asarray(predictions, dtype=float)
    rows["residual"] = pd.to_numeric(rows[target], errors="coerce") - rows["prediction"]
    rows["state"] = _geo_key(rows["state"])
    rows["state_sector"] = _state_sector(rows["state"], rows["sector"])
    return rows.groupby("canonical_project_id", as_index=False).agg(
        state=("state", "first"), state_sector=("state_sector", "first"), residual=("residual", "mean")
    )


def _prior(projects: pd.DataFrame, group: str, k: float, cap: float) -> dict[str, float]:
    eligible = projects[projects[group].ne(MISSING) & ~projects[group].str.startswith(MISSING + "||")]
    if eligible.empty:
        return {}
    stats = eligible.groupby(group).residual.agg(["mean", "count"])
    shrunk = stats["mean"] * stats["count"] / (stats["count"] + float(k))
    return {str(key): float(value) for key, value in shrunk.clip(-cap, cap).items()}


def corrections(frame: pd.DataFrame, state_prior: dict[str, float], state_sector_prior: dict[str, float], cap: float) -> np.ndarray:
    state = _geo_key(frame.get("state", pd.Series(None, index=frame.index)))
    ss = _state_sector(state, frame.get("sector", pd.Series(None, index=frame.index)))
    a = state.map(state_prior).fillna(0.0).to_numpy(dtype=float)
    b = ss.map(state_sector_prior).fillna(0.0).to_numpy(dtype=float)
    return np.clip(STATE_WEIGHT * a + STATE_SECTOR_WEIGHT * b, -cap, cap)


def _improvement(base: float, challenger: float) -> float:
    return (base - challenger) / base * 100.0 if base else 0.0


def _decision(cost_improvement: float, delay_improvement: float) -> str:
    return "PROMOTION CANDIDATE" if cost_improvement >= 0 and delay_improvement >= 0 and (cost_improvement > 0 or delay_improvement > 0) else "REGRESSION / DO NOT PROMOTE"


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    data, identity = build_training_dataset()
    temp_root = Path(tempfile.mkdtemp(prefix="sih-exp23-"))
    run_id = f"exp23-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    try:
        production = train_window_with_promoted_cost(training_start, training_end, test_end, data=data, identity=identity, artifact_root=temp_root)
        artifact_dir = temp_root / f"{training_start}_{training_end}"
        metadata = production["metadata"]
        contract = target_feature_contract(metadata)

        enriched = enrich_supervised_for_production(data.copy())
        enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
        if "state" not in enriched:
            enriched["state"] = None
        train, test = temporal_project_split(enriched, training_start, training_end, test_end)

        prod_cost = joblib.load(artifact_dir / "cost_model.pkl")
        prod_delay = joblib.load(artifact_dir / "delay_model.pkl")
        train_cost_pred = prod_cost.predict(train[contract["cost"]])
        train_delay_pred = np.maximum(0, prod_delay.predict(train[contract["delay"]]))
        base_cost_pred = prod_cost.predict(test[contract["cost"]])
        base_delay_pred = np.maximum(0, prod_delay.predict(test[contract["delay"]]))

        cost_projects = project_level_residuals(train, train_cost_pred, "actual_cost_overrun_percentage")
        delay_projects = project_level_residuals(train, train_delay_pred, "actual_delay_days")
        cost_state = _prior(cost_projects, "state", STATE_K, MAX_COST_CORRECTION)
        cost_ss = _prior(cost_projects, "state_sector", STATE_SECTOR_K, MAX_COST_CORRECTION)
        delay_state = _prior(delay_projects, "state", STATE_K, MAX_DELAY_CORRECTION)
        delay_ss = _prior(delay_projects, "state_sector", STATE_SECTOR_K, MAX_DELAY_CORRECTION)

        cost_correction = corrections(test, cost_state, cost_ss, MAX_COST_CORRECTION)
        delay_correction = corrections(test, delay_state, delay_ss, MAX_DELAY_CORRECTION)
        exp_cost_pred = base_cost_pred + cost_correction
        exp_delay_pred = np.maximum(0, base_delay_pred + delay_correction)

        base_cost = _regression_metrics(test.actual_cost_overrun_percentage, base_cost_pred, test.sample_weight, test.canonical_project_id)
        exp_cost = _regression_metrics(test.actual_cost_overrun_percentage, exp_cost_pred, test.sample_weight, test.canonical_project_id)
        base_delay = _regression_metrics(test.actual_delay_days, base_delay_pred, test.sample_weight, test.canonical_project_id)
        exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
        cost_gain = _improvement(float(base_cost["MAE"]), float(exp_cost["MAE"]))
        delay_gain = _improvement(float(base_delay["MAE"]), float(exp_delay["MAE"]))

        test_state = _geo_key(test["state"])
        known_geo = test_state.ne(MISSING)
        report = {
            "experiment": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "status": "complete", "decision": _decision(cost_gain, delay_gain),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_period": [training_start, training_end], "testing_period": [training_end + 1, test_end],
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "metrics": {
                "production_cost_mae": float(base_cost["MAE"]), "experiment_cost_mae": float(exp_cost["MAE"]), "cost_improvement_percentage": cost_gain,
                "production_delay_mae": float(base_delay["MAE"]), "experiment_delay_mae": float(exp_delay["MAE"]), "delay_improvement_percentage": delay_gain,
                "production_cost": base_cost, "experiment_cost": exp_cost, "production_delay": base_delay, "experiment_delay": exp_delay,
            },
            "method": {
                "state_shrinkage_k": STATE_K, "state_sector_shrinkage_k": STATE_SECTOR_K,
                "state_weight": STATE_WEIGHT, "state_sector_weight": STATE_SECTOR_WEIGHT,
                "max_cost_correction_pp": MAX_COST_CORRECTION, "max_delay_correction_days": MAX_DELAY_CORRECTION,
                "residual_unit": "one mean residual per training project",
                "holdout_targets_used_for_priors": False,
            },
            "coverage": {
                "training_projects": int(train.canonical_project_id.nunique()),
                "training_projects_with_state": int(cost_projects.state.ne(MISSING).sum()),
                "test_projects": int(test.canonical_project_id.nunique()), "test_snapshots": int(len(test)),
                "test_snapshot_state_share": float(known_geo.mean()),
                "cost_state_groups": len(cost_state), "cost_state_sector_groups": len(cost_ss),
                "delay_state_groups": len(delay_state), "delay_state_sector_groups": len(delay_ss),
                "cost_nonzero_correction_share": float(np.mean(np.abs(cost_correction) > 1e-12)),
                "delay_nonzero_correction_share": float(np.mean(np.abs(delay_correction) > 1e-12)),
            },
            "leakage_policy": "State and sector are same-snapshot administrative fields. Residual priors use only training-window targets, one residual per training project; future holdout targets never enter prior fitting. Missing/unseen geography gets zero correction.",
            "production_changed": False,
        }
        out = REPORT_ROOT / f"{training_start}_{training_end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        validation = test[["canonical_project_id", "project_name", "snapshot_date", "completion_year", "state", "sector", "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight"]].copy()
        validation["production_cost_prediction"] = base_cost_pred
        validation["geographic_cost_correction"] = cost_correction
        validation["experiment_cost_prediction"] = exp_cost_pred
        validation["production_delay_prediction"] = base_delay_pred
        validation["geographic_delay_correction"] = delay_correction
        validation["experiment_delay_prediction"] = exp_delay_pred
        validation.to_csv(out / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
