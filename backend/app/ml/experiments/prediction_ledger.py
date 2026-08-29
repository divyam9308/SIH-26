"""Canonical row-level evidence ledger for isolated ML experiments.

Future experiment PRs should persist the exact paired production/challenger
predictions used for headline MAE.  The ledger makes later slice analysis and
post-mortems reproducible without retraining the model or trusting a summary
number copied into a PR description.

The writer is deliberately experiment-only infrastructure.  It does not alter
production model artifacts, selection, or prediction behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from backend.app.ml.provenance import file_sha256, frame_fingerprint, git_commit_sha

ROOT = Path(__file__).resolve().parents[4]
LEDGER_SCHEMA_VERSION = 1
LEDGER_FILENAME = "prediction_ledger.csv"
LEDGER_MANIFEST_FILENAME = "prediction_ledger_manifest.json"
KEY_COLUMNS = ("canonical_project_id", "snapshot_date")
DEFAULT_SLICE_COLUMNS = (
    "completion_year",
    "lifecycle_stage",
    "sector",
    "implementing_agency",
    "state",
    "scale_bucket",
    "history_observations",
    "parser_family",
    "experiment_route",
)
TARGET_SPECS = {
    "cost": {
        "actual": "actual_cost_overrun_percentage",
        "production": "production_cost_prediction",
        "experiment": "experiment_cost_prediction",
    },
    "delay": {
        "actual": "actual_delay_days",
        "production": "production_delay_prediction",
        "experiment": "experiment_delay_prediction",
    },
}


def _prediction_array(values, *, rows: pd.DataFrame, name: str) -> np.ndarray:
    """Resolve either a source-column name or an array-like prediction."""
    if isinstance(values, str):
        if values not in rows:
            raise ValueError(f"prediction ledger source column is missing: {values}")
        values = rows[values]
    array = pd.to_numeric(pd.Series(values, index=rows.index), errors="coerce").to_numpy(dtype=float)
    if len(array) != len(rows):
        raise ValueError(f"{name} must have exactly {len(rows)} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return array


def _canonical_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*KEY_COLUMNS, "sample_weight"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("prediction ledger cohort missing columns: " + ", ".join(missing))
    cohort = frame[required].copy()
    cohort["canonical_project_id"] = cohort["canonical_project_id"].astype("string").str.strip()
    cohort["snapshot_date"] = pd.to_datetime(cohort["snapshot_date"], errors="coerce")
    cohort["sample_weight"] = pd.to_numeric(cohort["sample_weight"], errors="coerce")
    if cohort["canonical_project_id"].isna().any() or cohort["canonical_project_id"].eq("").any():
        raise ValueError("prediction ledger requires non-empty canonical_project_id")
    if cohort["snapshot_date"].isna().any():
        raise ValueError("prediction ledger contains invalid snapshot_date")
    weights = cohort["sample_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("prediction ledger sample_weight must be finite and non-negative")
    if cohort.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("prediction ledger contains duplicate project/snapshot keys")
    return cohort.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def cohort_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint only paired observation identity and post-filter weights."""
    return frame_fingerprint(_canonical_cohort(frame))


def validate_prediction_ledger(
    ledger: pd.DataFrame,
    *,
    require_project_balanced: bool = True,
    weight_tolerance: float = 1e-6,
) -> dict:
    """Validate ledger invariants and return compact diagnostics.

    Target-specific experiments may persist only Cost or only Delay, but every
    included target must contain actual, production, and challenger predictions.
    """
    if ledger.empty:
        raise ValueError("prediction ledger cannot be empty")
    cohort = _canonical_cohort(ledger)

    if "experiment_id" not in ledger or ledger["experiment_id"].astype(str).nunique() != 1:
        raise ValueError("prediction ledger must contain exactly one experiment_id")
    if "window" not in ledger or ledger["window"].astype(str).nunique() != 1:
        raise ValueError("prediction ledger must contain exactly one window")

    targets: list[str] = []
    for target, spec in TARGET_SPECS.items():
        present = [column in ledger for column in spec.values()]
        if any(present) and not all(present):
            missing = [column for column in spec.values() if column not in ledger]
            raise ValueError(f"prediction ledger {target} target is incomplete: {', '.join(missing)}")
        if all(present):
            targets.append(target)
            for column in spec.values():
                values = pd.to_numeric(ledger[column], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"prediction ledger column {column} contains missing/non-finite values")
    if not targets:
        raise ValueError("prediction ledger must include at least one complete Cost or Delay target")

    weight_sums = cohort.groupby("canonical_project_id", sort=False)["sample_weight"].sum()
    if require_project_balanced and not np.allclose(
        weight_sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=float(weight_tolerance)
    ):
        raise ValueError(
            "prediction ledger weights must be recalculated after filtering so each project has total weight 1"
        )

    return {
        "projects": int(cohort["canonical_project_id"].nunique()),
        "snapshots": int(len(cohort)),
        "targets": targets,
        "project_weight_sum_min": float(weight_sums.min()),
        "project_weight_sum_max": float(weight_sums.max()),
        "cohort_fingerprint": frame_fingerprint(cohort),
    }


def build_prediction_ledger(
    rows: pd.DataFrame,
    *,
    experiment_id: str,
    window: str,
    production_cost_prediction=None,
    experiment_cost_prediction=None,
    production_delay_prediction=None,
    experiment_delay_prediction=None,
    actual_cost_column: str = "actual_cost_overrun_percentage",
    actual_delay_column: str = "actual_delay_days",
    extra_columns: Sequence[str] | None = None,
    require_project_balanced: bool = True,
) -> pd.DataFrame:
    """Build the canonical paired row-level evidence table.

    Predictions can be arrays/Series or names of columns already present in
    ``rows``.  The caller must pass both production and experiment predictions
    for each target it wants in the ledger.  This prevents a target-specific
    challenger from accidentally fabricating evidence for an unchanged target.
    """
    if rows.empty:
        raise ValueError("prediction ledger source rows cannot be empty")
    if not str(experiment_id).strip() or not str(window).strip():
        raise ValueError("prediction ledger requires experiment_id and window")

    cohort = _canonical_cohort(rows)
    ordered = rows.copy()
    ordered["canonical_project_id"] = ordered["canonical_project_id"].astype("string").str.strip()
    ordered["snapshot_date"] = pd.to_datetime(ordered["snapshot_date"], errors="coerce")
    ordered["sample_weight"] = pd.to_numeric(ordered["sample_weight"], errors="coerce")
    ordered = ordered.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)

    ledger = cohort.copy()
    ledger.insert(0, "window", str(window))
    ledger.insert(0, "experiment_id", str(experiment_id))

    requested_extra = tuple(extra_columns) if extra_columns is not None else DEFAULT_SLICE_COLUMNS
    for column in requested_extra:
        if column in ordered and column not in ledger:
            ledger[column] = ordered[column].to_numpy(copy=True)

    cost_requested = production_cost_prediction is not None or experiment_cost_prediction is not None
    if cost_requested:
        if production_cost_prediction is None or experiment_cost_prediction is None:
            raise ValueError("Cost ledger requires both production and experiment predictions")
        if actual_cost_column not in ordered:
            raise ValueError(f"Cost ledger actual column is missing: {actual_cost_column}")
        ledger[TARGET_SPECS["cost"]["actual"]] = _prediction_array(
            ordered[actual_cost_column], rows=ordered, name=actual_cost_column
        )
        ledger[TARGET_SPECS["cost"]["production"]] = _prediction_array(
            production_cost_prediction, rows=ordered, name="production_cost_prediction"
        )
        ledger[TARGET_SPECS["cost"]["experiment"]] = _prediction_array(
            experiment_cost_prediction, rows=ordered, name="experiment_cost_prediction"
        )
        ledger["production_cost_abs_error"] = (
            ledger["production_cost_prediction"] - ledger["actual_cost_overrun_percentage"]
        ).abs()
        ledger["experiment_cost_abs_error"] = (
            ledger["experiment_cost_prediction"] - ledger["actual_cost_overrun_percentage"]
        ).abs()
        ledger["cost_abs_error_improvement"] = (
            ledger["production_cost_abs_error"] - ledger["experiment_cost_abs_error"]
        )

    delay_requested = production_delay_prediction is not None or experiment_delay_prediction is not None
    if delay_requested:
        if production_delay_prediction is None or experiment_delay_prediction is None:
            raise ValueError("Delay ledger requires both production and experiment predictions")
        if actual_delay_column not in ordered:
            raise ValueError(f"Delay ledger actual column is missing: {actual_delay_column}")
        ledger[TARGET_SPECS["delay"]["actual"]] = _prediction_array(
            ordered[actual_delay_column], rows=ordered, name=actual_delay_column
        )
        ledger[TARGET_SPECS["delay"]["production"]] = _prediction_array(
            production_delay_prediction, rows=ordered, name="production_delay_prediction"
        )
        ledger[TARGET_SPECS["delay"]["experiment"]] = _prediction_array(
            experiment_delay_prediction, rows=ordered, name="experiment_delay_prediction"
        )
        ledger["production_delay_abs_error"] = (
            ledger["production_delay_prediction"] - ledger["actual_delay_days"]
        ).abs()
        ledger["experiment_delay_abs_error"] = (
            ledger["experiment_delay_prediction"] - ledger["actual_delay_days"]
        ).abs()
        ledger["delay_abs_error_improvement"] = (
            ledger["production_delay_abs_error"] - ledger["experiment_delay_abs_error"]
        )

    validate_prediction_ledger(ledger, require_project_balanced=require_project_balanced)
    return ledger


def _target_metrics(ledger: pd.DataFrame, target: str) -> dict:
    spec = TARGET_SPECS[target]
    production_error = (ledger[spec["production"]] - ledger[spec["actual"]]).abs()
    experiment_error = (ledger[spec["experiment"]] - ledger[spec["actual"]]).abs()
    weights = pd.to_numeric(ledger["sample_weight"], errors="raise").to_numpy(dtype=float)
    production_mae = float(np.average(production_error.to_numpy(dtype=float), weights=weights))
    experiment_mae = float(np.average(experiment_error.to_numpy(dtype=float), weights=weights))
    improvement = production_mae - experiment_mae
    return {
        "production_mae": production_mae,
        "experiment_mae": experiment_mae,
        "absolute_mae_improvement": improvement,
        "percentage_mae_improvement": (improvement / production_mae * 100.0) if production_mae else None,
    }


def prediction_ledger_manifest(ledger: pd.DataFrame) -> dict:
    diagnostics = validate_prediction_ledger(ledger)
    targets = list(diagnostics["targets"])
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "experiment_id": str(ledger["experiment_id"].iloc[0]),
        "window": str(ledger["window"].iloc[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit_sha(ROOT),
        "projects": diagnostics["projects"],
        "snapshots": diagnostics["snapshots"],
        "targets": targets,
        "project_weight_sum_min": diagnostics["project_weight_sum_min"],
        "project_weight_sum_max": diagnostics["project_weight_sum_max"],
        "cohort_fingerprint": diagnostics["cohort_fingerprint"],
        "ledger_fingerprint": frame_fingerprint(ledger),
        "columns": list(ledger.columns),
        "slice_columns_present": [column for column in DEFAULT_SLICE_COLUMNS if column in ledger],
        "metrics": {target: _target_metrics(ledger, target) for target in targets},
        "semantics": {
            "positive_row_error_improvement": "production absolute error minus experiment absolute error; positive favors experiment",
            "weighting": "sample_weight must sum to 1 within each project after all comparison filtering",
            "cohort_identity": "canonical project id + snapshot date + post-filter sample weight",
        },
    }


def write_prediction_ledger(
    ledger: pd.DataFrame,
    directory: Path,
    *,
    extra_manifest: dict | None = None,
) -> dict:
    """Atomically persist canonical CSV + manifest and return their metadata."""
    validate_prediction_ledger(ledger)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / LEDGER_FILENAME
    manifest_path = directory / LEDGER_MANIFEST_FILENAME

    csv_tmp = directory / f".{LEDGER_FILENAME}.tmp"
    ledger.to_csv(
        csv_tmp,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        na_rep="<NA>",
        float_format="%.12g",
        lineterminator="\n",
    )
    csv_tmp.replace(ledger_path)

    manifest = prediction_ledger_manifest(ledger)
    if extra_manifest:
        protected = {
            "schema_version",
            "experiment_id",
            "window",
            "projects",
            "snapshots",
            "targets",
            "cohort_fingerprint",
            "ledger_fingerprint",
            "metrics",
        }
        overlap = protected.intersection(extra_manifest)
        if overlap:
            raise ValueError("extra prediction-ledger manifest cannot override: " + ", ".join(sorted(overlap)))
        manifest.update(extra_manifest)
    manifest["ledger_file"] = LEDGER_FILENAME
    manifest["ledger_file_sha256"] = file_sha256(ledger_path)

    json_tmp = directory / f".{LEDGER_MANIFEST_FILENAME}.tmp"
    json_tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    json_tmp.replace(manifest_path)
    return {
        "ledger_path": ledger_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def write_experiment_prediction_ledger(
    ledger: pd.DataFrame,
    *,
    experiment_id: str,
    window: str,
    run_id: str,
    extra_manifest: dict | None = None,
) -> dict:
    """Persist under the immutable experiment model root, never production."""
    from backend.app.ml.experiments.framework import experiment_run_directory

    destination = experiment_run_directory(experiment_id, window, run_id)
    return write_prediction_ledger(ledger, destination, extra_manifest=extra_manifest)


def assert_prediction_ledger_matches_cohort(ledger: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Reject ledgers that do not represent the exact scored observation cohort."""
    actual = cohort_fingerprint(ledger)
    expected = cohort_fingerprint(rows)
    if actual != expected:
        raise ValueError(
            f"prediction ledger cohort mismatch: ledger={actual} expected={expected}"
        )
