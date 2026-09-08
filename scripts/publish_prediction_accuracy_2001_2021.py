"""Retrain the production 2001-2021 lifecycle model and publish UI evidence.

This script is intentionally fixed to the canonical 2001-2021 training window and
the independently evaluated 2023-2025 cohort used by the Prediction Accuracy page.
It refuses to publish partial, stale, or empty chart evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.monthly_training import MODEL_ROOT, _regression_metrics, _risk_metrics
from backend.app.services.lifecycle_retraining_service import _write_evaluation_reports, _write_run_manifest, retrain_lifecycle

TRAIN_START = 2001
TRAIN_END = 2021
TEST_YEARS = (2023, 2024, 2025)
TARGET = MODEL_ROOT / f"{TRAIN_START}_{TRAIN_END}"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Prediction evidence is missing required columns: {missing}")


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Prediction evidence contains non-finite values in {column}")
    return values


def _annual_holdout_metrics(frame: pd.DataFrame) -> dict:
    _require_columns(
        frame,
        (
            "canonical_project_id",
            "completion_year",
            "actual_cost_overrun_percentage",
            "predicted_cost_overrun",
            "actual_delay_days",
            "predicted_delay_days",
            "actual_risk",
            "predicted_risk",
            "sample_weight",
        ),
    )
    years = pd.to_numeric(frame["completion_year"], errors="coerce")
    observed = tuple(sorted(int(year) for year in years.dropna().unique()))
    if observed != TEST_YEARS:
        raise RuntimeError(f"Expected holdout years {list(TEST_YEARS)}, found {list(observed)}")

    for column in (
        "actual_cost_overrun_percentage",
        "predicted_cost_overrun",
        "actual_delay_days",
        "predicted_delay_days",
        "sample_weight",
    ):
        _finite_series(frame, column)

    folds = []
    for year in TEST_YEARS:
        part = frame.loc[years.eq(year)].copy()
        if part.empty or part["canonical_project_id"].nunique() < 1:
            raise RuntimeError(f"Holdout year {year} has no project evidence")
        cost = _regression_metrics(
            part["actual_cost_overrun_percentage"],
            pd.to_numeric(part["predicted_cost_overrun"], errors="raise").to_numpy(float),
            part["sample_weight"],
            part["canonical_project_id"],
        )
        delay = _regression_metrics(
            part["actual_delay_days"],
            pd.to_numeric(part["predicted_delay_days"], errors="raise").to_numpy(float),
            part["sample_weight"],
            part["canonical_project_id"],
        )
        risk = _risk_metrics(part["actual_risk"], part["predicted_risk"].to_numpy(), part["sample_weight"])
        folds.append(
            {
                "test_year": year,
                "cost_MAE": float(cost["MAE"]),
                "delay_MAE_days": float(delay["MAE"]),
                "risk_f1": float(risk["macro_f1"]),
                "test_projects": int(part["canonical_project_id"].nunique()),
                "test_snapshots": int(len(part)),
            }
        )

    payload = {
        "model_version": "2001_2021",
        "training_period": [TRAIN_START, TRAIN_END],
        "testing_period": [TEST_YEARS[0], TEST_YEARS[-1]],
        "policy": "One fixed 2001-2021 production model; metrics grouped by true completion year within the untouched 2022-2025 holdout.",
        "status": "generated_from_canonical_prediction_validation",
        "fold_count": len(folds),
        "folds": folds,
    }
    return payload


def _update_published_evaluation(frame: pd.DataFrame, result: dict) -> None:
    cost = _regression_metrics(
        frame["actual_cost_overrun_percentage"],
        pd.to_numeric(frame["predicted_cost_overrun"], errors="raise").to_numpy(float),
        frame["sample_weight"],
        frame["canonical_project_id"],
    )
    delay = _regression_metrics(
        frame["actual_delay_days"],
        pd.to_numeric(frame["predicted_delay_days"], errors="raise").to_numpy(float),
        frame["sample_weight"],
        frame["canonical_project_id"],
    )
    risk = _risk_metrics(frame["actual_risk"], frame["predicted_risk"].to_numpy(), frame["sample_weight"])
    evaluation_path = TARGET / "evaluation_results.json"
    evaluation = json.loads(evaluation_path.read_text())
    metadata = dict(evaluation["metadata"])
    metadata.update({
        "testing_period": [TEST_YEARS[0], TEST_YEARS[-1]],
        "test_start": TEST_YEARS[0],
        "test_end": TEST_YEARS[-1],
        "evaluated_test_start": TEST_YEARS[0],
        "evaluated_test_end": TEST_YEARS[-1],
        "unique_test_projects": int(frame["canonical_project_id"].nunique()),
        "test_snapshots": int(len(frame)),
    })
    evaluation["metadata"] = metadata
    evaluation["lifecycle"]["metrics"]["cost"].update(cost)
    evaluation["lifecycle"]["metrics"]["delay"].update(delay)
    evaluation["lifecycle"]["metrics"]["risk"].update(risk)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
    (TARGET / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    result["metadata"] = metadata
    result["lifecycle"]["metrics"] = evaluation["lifecycle"]["metrics"]
    _write_evaluation_reports(result, TARGET)
    _write_run_manifest(TRAIN_START, TRAIN_END, result, TARGET)


def publish() -> dict:
    result = retrain_lifecycle(TRAIN_START, TRAIN_END)
    metadata = result.get("metadata") or {}
    training = tuple(metadata.get("training_period") or ())
    testing = tuple(metadata.get("testing_period") or ())
    if training != (TRAIN_START, TRAIN_END):
        raise RuntimeError(f"Retrain published wrong training period: {training}")
    if testing != (TRAIN_END + 1, 2025):
        raise RuntimeError(f"Retrain published wrong source holdout: {testing}")

    validation_path = TARGET / "prediction_validation.csv"
    evaluation_path = TARGET / "evaluation_results.json"
    if not validation_path.exists() or validation_path.stat().st_size == 0:
        raise RuntimeError("Retrain did not publish prediction_validation.csv")
    if not evaluation_path.exists() or evaluation_path.stat().st_size == 0:
        raise RuntimeError("Retrain did not publish evaluation_results.json")

    frame = pd.read_csv(validation_path, dtype={"canonical_project_id": str})
    frame = frame.loc[pd.to_numeric(frame["completion_year"], errors="coerce").isin(TEST_YEARS)].copy()
    if frame.empty:
        raise RuntimeError("The requested 2023-2025 prediction validation ledger is empty")
    frame.to_csv(validation_path, index=False)
    _update_published_evaluation(frame, result)
    rolling = _annual_holdout_metrics(frame)
    rolling_path = TARGET / "rolling_validation_results.json"
    rolling_path.write_text(json.dumps(rolling, indent=2, allow_nan=False) + "\n")

    summary = {
        "training_period": [TRAIN_START, TRAIN_END],
        "testing_period": [TEST_YEARS[0], TEST_YEARS[-1]],
        "validation_rows": int(len(frame)),
        "validation_projects": int(frame["canonical_project_id"].nunique()),
        "annual_folds": len(rolling["folds"]),
        "artifacts": {
            "evaluation_results": str(evaluation_path),
            "prediction_validation": str(validation_path),
            "rolling_validation": str(rolling_path),
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    publish()
