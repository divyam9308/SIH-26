from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.app.ml.monthly_lifecycle import build_training_dataset

TRAIN_START = 2001
TRAIN_END = 2021
TEST_START = 2022
TEST_END = 2025


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _weighted_metrics(rows: pd.DataFrame, actual_col: str, prediction_col: str) -> dict:
    actual = pd.to_numeric(rows[actual_col], errors="coerce")
    predicted = pd.to_numeric(rows[prediction_col], errors="coerce")
    weights = pd.to_numeric(rows.get("sample_weight", 1.0), errors="coerce").fillna(1.0)
    mask = actual.notna() & predicted.notna() & np.isfinite(predicted) & weights.gt(0)
    actual = actual[mask].to_numpy(dtype=float)
    predicted = predicted[mask].to_numpy(dtype=float)
    weights = weights[mask].to_numpy(dtype=float)
    if len(actual) == 0:
        return {"MAE": None, "RMSE": None, "R2": None, "rows": 0, "unique_projects": 0}
    project_count = int(rows.loc[mask, "canonical_project_id"].astype("string").nunique())
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted, sample_weight=weights)), 3),
        "RMSE": round(float(math.sqrt(mean_squared_error(actual, predicted, sample_weight=weights))), 3),
        "R2": round(float(r2_score(actual, predicted, sample_weight=weights)), 4) if len(actual) > 1 else None,
        "rows": int(len(actual)),
        "unique_projects": project_count,
    }


def _project_outcomes(training: pd.DataFrame) -> pd.DataFrame:
    frame = training.copy()
    frame["completion_year"] = pd.to_numeric(frame["completion_year"], errors="coerce")
    frame = frame[frame["completion_year"].between(TRAIN_START, TRAIN_END)].copy()
    columns = ["canonical_project_id", "actual_cost_overrun_percentage", "actual_delay_days"]
    frame = frame[columns].dropna(subset=["canonical_project_id"]).copy()
    for target in ("actual_cost_overrun_percentage", "actual_delay_days"):
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    # Outcomes are project-level labels repeated over snapshots. Median protects the
    # threshold derivation against accidental duplicate-row inconsistencies.
    return frame.groupby("canonical_project_id", as_index=False).agg(
        actual_cost_overrun_percentage=("actual_cost_overrun_percentage", "median"),
        actual_delay_days=("actual_delay_days", "median"),
    )


def derive_training_thresholds(training: pd.DataFrame) -> dict:
    projects = _project_outcomes(training)
    result = {}
    for name, column in (("cost", "actual_cost_overrun_percentage"), ("delay", "actual_delay_days")):
        values = pd.to_numeric(projects[column], errors="coerce").dropna()
        if len(values) < 20:
            raise ValueError(f"Insufficient training projects to derive {name} tail thresholds: {len(values)}")
        result[name] = {
            "p90": _finite(values.quantile(0.90)),
            "p95": _finite(values.quantile(0.95)),
            "p99": _finite(values.quantile(0.99)),
            "training_projects": int(len(values)),
            "source": "2001-2021 unique-project actual outcome distribution",
        }
    return result


def _bands(values: pd.Series, thresholds: dict) -> dict[str, pd.Series]:
    v = pd.to_numeric(values, errors="coerce")
    p90, p95, p99 = thresholds["p90"], thresholds["p95"], thresholds["p99"]
    return {
        "all": v.notna(),
        "normal_le_p90": v.notna() & v.le(p90),
        "tail_p90_p95": v.gt(p90) & v.le(p95),
        "extreme_p95_p99": v.gt(p95) & v.le(p99),
        "ultra_tail_gt_p99": v.gt(p99),
        "excluding_top_5pct_le_p95": v.notna() & v.le(p95),
        "excluding_top_1pct_le_p99": v.notna() & v.le(p99),
        "tail_gt_p95": v.gt(p95),
    }


def _target_report(ledger: pd.DataFrame, *, actual_col: str, prediction_col: str, thresholds: dict) -> dict:
    bands = _bands(ledger[actual_col], thresholds)
    return {
        name: _weighted_metrics(ledger.loc[mask].copy(), actual_col, prediction_col)
        for name, mask in bands.items()
    }


def _delta(full: dict, subset: dict) -> dict:
    out = {}
    for key in ("MAE", "RMSE", "R2"):
        a, b = full.get(key), subset.get(key)
        out[key] = round(float(b) - float(a), 4) if a is not None and b is not None else None
    return out


def build_report(ledger: pd.DataFrame, training: pd.DataFrame) -> dict:
    required = {
        "canonical_project_id",
        "actual_cost_overrun_percentage",
        "predicted_cost_overrun",
        "actual_delay_days",
        "predicted_delay_days",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise KeyError(f"Prediction ledger missing required columns: {missing}")

    thresholds = derive_training_thresholds(training)
    cost = _target_report(
        ledger,
        actual_col="actual_cost_overrun_percentage",
        prediction_col="predicted_cost_overrun",
        thresholds=thresholds["cost"],
    )
    delay = _target_report(
        ledger,
        actual_col="actual_delay_days",
        prediction_col="predicted_delay_days",
        thresholds=thresholds["delay"],
    )

    return {
        "experiment_id": "tail_sensitivity_2001_2021",
        "model_role": "diagnostic_only",
        "window": {
            "training_start": TRAIN_START,
            "training_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
        },
        "leakage_policy": (
            "Tail thresholds are derived only from unique-project 2001-2021 training outcomes. "
            "No 2022-2025 prediction errors are used to select or define exclusions. "
            "The full holdout remains the primary official evaluation; subset metrics are diagnostics only."
        ),
        "thresholds": thresholds,
        "cost": {
            "bands": cost,
            "impact_vs_full": {
                "excluding_top_5pct": _delta(cost["all"], cost["excluding_top_5pct_le_p95"]),
                "excluding_top_1pct": _delta(cost["all"], cost["excluding_top_1pct_le_p99"]),
            },
        },
        "delay": {
            "bands": delay,
            "impact_vs_full": {
                "excluding_top_5pct": _delta(delay["all"], delay["excluding_top_5pct_le_p95"]),
                "excluding_top_1pct": _delta(delay["all"], delay["excluding_top_1pct_le_p99"]),
            },
        },
        "interpretation": {
            "positive_r2_delta_when_excluding_tail": "headline R2 increases after removing the training-defined tail",
            "negative_mae_or_rmse_delta_when_excluding_tail": "headline absolute/squared error falls after removing the training-defined tail",
            "promotion_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("models/monthly_lifecycle/2001_2021/prediction_validation.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test-output/tail-sensitivity-2001-2021/tail_sensitivity_result.json"),
    )
    args = parser.parse_args()

    if not args.ledger.exists():
        raise FileNotFoundError(
            f"Frozen 2001-2021 prediction ledger is required at {args.ledger}. "
            "Rebuild the canonical reference window first; do not substitute a legacy evaluation."
        )

    ledger = pd.read_csv(args.ledger, dtype={"canonical_project_id": str})
    training, _identity = build_training_dataset()
    report = build_report(ledger, training)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
