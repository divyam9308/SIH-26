"""Experiment 7: leakage-safe hierarchical agency + sector residual correction.

The experiment leaves the production 25-feature pipeline unchanged. It trains the
current promoted production cost baseline, then learns a small additive correction
from training-period residuals only. Corrections are hierarchical and shrunk toward
zero so sparse agencies/sectors cannot dominate predictions.
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
from backend.app.ml.production_cost_baseline import (
    enrich_supervised_for_production,
    target_feature_contract,
    train_window_with_promoted_cost,
)
from backend.app.ml.experiments.registry import record_experiment

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "reports" / "experiments" / "exp_7"
EXPERIMENT_ID = "exp_7"
EXPERIMENT_NAME = "Hierarchical agency + sector residual correction / historical-performance priors"

# Conservative empirical-Bayes style shrinkage constants. A group with n examples
# receives weight n/(n+k); sparse groups therefore stay close to zero correction.
AGENCY_K = 12.0
SECTOR_K = 20.0
AGENCY_WEIGHT = 0.65
SECTOR_WEIGHT = 0.35
MAX_ABS_CORRECTION = 15.0


def _safe_group_key(series: pd.Series) -> pd.Series:
    return series.fillna("__missing__").astype(str).str.strip().replace("", "__missing__")


def _project_level_residuals(train: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    rows = train[["canonical_project_id", "implementing_agency", "sector", "actual_cost_overrun_percentage"]].copy()
    rows["predicted_cost"] = np.asarray(predictions, dtype=float)
    rows["residual"] = rows.actual_cost_overrun_percentage - rows.predicted_cost
    rows["implementing_agency"] = _safe_group_key(rows.implementing_agency)
    rows["sector"] = _safe_group_key(rows.sector)
    # Prevent projects with many monthly snapshots from dominating the correction prior.
    return (
        rows.groupby("canonical_project_id", as_index=False)
        .agg(
            implementing_agency=("implementing_agency", "first"),
            sector=("sector", "first"),
            residual=("residual", "mean"),
        )
    )


def _group_prior(project_residuals: pd.DataFrame, column: str, k: float) -> dict[str, float]:
    stats = project_residuals.groupby(column).residual.agg(["mean", "count"])
    shrink = stats["count"] / (stats["count"] + float(k))
    values = (stats["mean"] * shrink).clip(-MAX_ABS_CORRECTION, MAX_ABS_CORRECTION)
    return {str(index): float(value) for index, value in values.items()}


def _corrections(frame: pd.DataFrame, agency_prior: dict[str, float], sector_prior: dict[str, float]) -> np.ndarray:
    agencies = _safe_group_key(frame.implementing_agency)
    sectors = _safe_group_key(frame.sector)
    agency = agencies.map(agency_prior).fillna(0.0).to_numpy(dtype=float)
    sector = sectors.map(sector_prior).fillna(0.0).to_numpy(dtype=float)
    return np.clip(AGENCY_WEIGHT * agency + SECTOR_WEIGHT * sector, -MAX_ABS_CORRECTION, MAX_ABS_CORRECTION)


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")

    run_id = f"exp7-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    temp_root = Path(tempfile.mkdtemp(prefix="sih-exp7-"))
    try:
        baseline = train_window_with_promoted_cost(
            training_start,
            training_end,
            test_end,
            data=data,
            identity=identity,
            artifact_root=temp_root,
        )
        artifact_dir = temp_root / f"{training_start}_{training_end}"
        metadata = baseline["metadata"]
        contract = target_feature_contract(metadata)
        cost_features = contract["cost"]

        enriched = enrich_supervised_for_production(data)
        enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
        train, test = temporal_project_split(enriched, training_start, training_end, test_end)

        model = joblib.load(artifact_dir / "cost_model.pkl")
        train_predictions = model.predict(train[cost_features])
        test_predictions = model.predict(test[cost_features])

        project_residuals = _project_level_residuals(train, train_predictions)
        agency_prior = _group_prior(project_residuals, "implementing_agency", AGENCY_K)
        sector_prior = _group_prior(project_residuals, "sector", SECTOR_K)
        correction = _corrections(test, agency_prior, sector_prior)
        corrected_predictions = test_predictions + correction

        baseline_metrics = _regression_metrics(
            test.actual_cost_overrun_percentage,
            test_predictions,
            test.sample_weight,
            test.canonical_project_id,
        )
        corrected_metrics = _regression_metrics(
            test.actual_cost_overrun_percentage,
            corrected_predictions,
            test.sample_weight,
            test.canonical_project_id,
        )

        base_mae = float(baseline_metrics["MAE"])
        exp_mae = float(corrected_metrics["MAE"])
        improvement_pct = (base_mae - exp_mae) / base_mae * 100.0 if base_mae else 0.0
        decision = "ACCEPTED" if exp_mae < base_mae else "REJECTED"

        validation = test[[
            "canonical_project_id", "project_name", "snapshot_date", "completion_year",
            "implementing_agency", "sector", "actual_cost_overrun_percentage", "sample_weight"
        ]].copy()
        validation["baseline_prediction"] = test_predictions
        validation["hierarchical_correction"] = correction
        validation["experiment_prediction"] = corrected_predictions
        validation["baseline_abs_error"] = np.abs(validation.baseline_prediction - validation.actual_cost_overrun_percentage)
        validation["experiment_abs_error"] = np.abs(validation.experiment_prediction - validation.actual_cost_overrun_percentage)

        report = {
            "experiment": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "run_id": run_id,
            "status": "complete",
            "decision": decision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_period": [training_start, training_end],
            "testing_period": [training_end + 1, test_end],
            "production_baseline": metadata.get("production_cost_baseline"),
            "cost_feature_count": len(cost_features),
            "cost_features": cost_features,
            "method": {
                "type": "additive residual correction",
                "agency_shrinkage_k": AGENCY_K,
                "sector_shrinkage_k": SECTOR_K,
                "agency_weight": AGENCY_WEIGHT,
                "sector_weight": SECTOR_WEIGHT,
                "max_abs_correction_percentage_points": MAX_ABS_CORRECTION,
                "residual_unit": "one mean residual per training project",
                "holdout_used_for_prior_fitting": False,
            },
            "metrics": {
                "baseline_cost_mae": base_mae,
                "experiment_cost_mae": exp_mae,
                "absolute_mae_change": exp_mae - base_mae,
                "improvement_percentage": improvement_pct,
                "baseline": baseline_metrics,
                "experiment": corrected_metrics,
            },
            "coverage": {
                "training_projects_for_priors": int(project_residuals.canonical_project_id.nunique()),
                "agency_groups": len(agency_prior),
                "sector_groups": len(sector_prior),
                "holdout_snapshots": int(len(test)),
                "holdout_projects": int(test.canonical_project_id.nunique()),
                "nonzero_correction_share": float(np.mean(np.abs(correction) > 1e-12)),
                "mean_abs_correction": float(np.mean(np.abs(correction))),
            },
            "leakage_policy": (
                "The 25-feature production baseline is unchanged. Agency/sector correction priors are fit only from "
                "projects in the selected training completion-year window; future holdout targets never enter prior fitting."
            ),
        }

        out_dir = REPORT_ROOT / f"{training_start}_{training_end}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        validation.to_csv(out_dir / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
        (out_dir / "agency_priors.json").write_text(json.dumps(agency_prior, indent=2, allow_nan=False) + "\n")
        (out_dir / "sector_priors.json").write_text(json.dumps(sector_prior, indent=2, allow_nan=False) + "\n")

        record_experiment({
            "experiment_id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "run_id": run_id,
            "status": "complete",
            "decision": decision,
            "created_at": report["created_at"],
            "training_period": report["training_period"],
            "testing_period": report["testing_period"],
            "baseline_cost_mae": base_mae,
            "experiment_cost_mae": exp_mae,
            "improvement_percentage": improvement_pct,
            "production_changed": False,
        })
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
