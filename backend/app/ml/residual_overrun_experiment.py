"""Experiment 3: remaining-overrun forecasting on the exact direct-model cohort.

This module is intentionally isolated from the production lifecycle model. It
answers one question only: does predicting the *remaining* cost deterioration
and reconstructing final overrun outperform predicting final overrun directly,
when both approaches see the same features and the exact same train/test
snapshots?
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import BASELINE_FEATURES, CANDIDATE_FEATURES, build_training_dataset
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    _select_regressor,
    temporal_project_split,
)

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports" / "experiments"
CURRENT_OVERRUN = "cost_escalation_percentage"
FINAL_TARGET = "actual_cost_overrun_percentage"
RESIDUAL_TARGET = "remaining_cost_overrun_percentage"


def reconstruct_final_overrun(current_overrun, predicted_remaining) -> np.ndarray:
    """Reconstruct the user-facing final-overrun forecast."""
    return np.asarray(current_overrun, dtype=float) + np.asarray(predicted_remaining, dtype=float)


def _with_residual_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[RESIDUAL_TARGET] = pd.to_numeric(result[FINAL_TARGET], errors="coerce") - pd.to_numeric(result[CURRENT_OVERRUN], errors="coerce")
    return result


def prepare_common_cost_cohort(data: pd.DataFrame, training_start: int, training_end: int, test_end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one common cohort used by both direct and residual approaches.

    Rows lacking current cost escalation cannot define a residual target, so they
    are removed *before* either model is fit/evaluated. This prevents the common
    mistake of comparing residual and direct metrics on different snapshots.
    """
    frame = data.copy()
    frame["completion_year"] = pd.to_numeric(frame["completion_year"], errors="coerce")
    train, test = temporal_project_split(frame, int(training_start), int(training_end), int(test_end))
    required = [FINAL_TARGET, CURRENT_OVERRUN, "sample_weight", "canonical_project_id"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Residual-overrun experiment requires columns: {', '.join(missing)}")
    train = train.dropna(subset=[FINAL_TARGET, CURRENT_OVERRUN]).copy()
    test = test.dropna(subset=[FINAL_TARGET, CURRENT_OVERRUN]).copy()
    if train.canonical_project_id.nunique() < 10 or test.canonical_project_id.nunique() < 2:
        raise ValueError(
            "Insufficient common direct/residual lifecycle cohort after requiring current cost escalation: "
            f"train projects={train.canonical_project_id.nunique()}, test projects={test.canonical_project_id.nunique()}"
        )
    return _with_residual_target(train), _with_residual_target(test)


def _cohort_digest(frame: pd.DataFrame) -> str:
    columns = [name for name in ("canonical_project_id", "snapshot_date", "completion_year") if name in frame]
    stable = frame[columns].copy() if columns else frame.index.to_frame(index=False)
    if columns:
        stable = stable.sort_values(columns, kind="mergesort", na_position="last")
    canonical = stable.astype("string").fillna("<NA>")
    raw = pd.util.hash_pandas_object(canonical, index=False, categorize=True).to_numpy().tobytes()
    return "sha256:" + sha256(raw).hexdigest()


def _stage_cost_metrics(rows: pd.DataFrame, prediction_column: str) -> dict:
    result: dict[str, dict] = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = rows[rows.lifecycle_stage.eq(stage)] if "lifecycle_stage" in rows else rows.iloc[0:0]
        if part.empty:
            result[stage] = {"available": False, "reason": "no common test snapshots for this lifecycle stage"}
            continue
        result[stage] = {
            "available": True,
            "cost": _regression_metrics(
                part[FINAL_TARGET],
                part[prediction_column].to_numpy(dtype=float),
                part.sample_weight,
                part.canonical_project_id,
            ),
        }
    return result


def run_residual_overrun_experiment(
    training_start: int,
    training_end: int,
    test_end: int | None = None,
    *,
    data: pd.DataFrame | None = None,
    persist: bool = True,
) -> dict:
    """Compare direct vs remaining-overrun forecasting on identical snapshots.

    The same audited features and the same selected regressor family are used for
    both approaches. This deliberately isolates the target formulation rather
    than allowing a different algorithm or cohort to create a false improvement.
    """
    if data is None:
        data, _identity = build_training_dataset()
    frame = data.copy()
    frame["completion_year"] = pd.to_numeric(frame["completion_year"], errors="coerce")
    years = frame.completion_year.dropna().astype(int)
    if years.empty:
        raise ValueError("No identity-verified lifecycle completion years are available.")
    maximum_year = int(years.max())
    test_end = maximum_year if test_end is None else int(test_end)
    if int(training_end) >= test_end:
        raise ValueError("Residual experiment requires at least one future holdout year after the training cutoff.")

    train, test = prepare_common_cost_cohort(frame, int(training_start), int(training_end), test_end)
    audit = audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        safely_as_of_features=set(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "observed as-of snapshot; potential late-stage target proxy",
            "cost_escalation_percentage": "observed current escalation and explicit residual anchor",
        },
    )
    features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))

    # Select one algorithm on the direct target using only internal temporal
    # validation, then use that exact algorithm family for the residual target.
    # This makes the primary comparison a target-formulation experiment.
    algorithm, direct_model, direct_internal = _select_regressor(train, features, FINAL_TARGET, 27103)
    residual_model = _fit_pipeline(_regressors(27103)[algorithm], train, features, RESIDUAL_TARGET)

    direct_final = np.asarray(direct_model.predict(test[features]), dtype=float)
    residual_remaining = np.asarray(residual_model.predict(test[features]), dtype=float)
    reconstructed_final = reconstruct_final_overrun(test[CURRENT_OVERRUN], residual_remaining)

    direct_metrics = _regression_metrics(FINAL_TARGET and test[FINAL_TARGET], direct_final, test.sample_weight, test.canonical_project_id)
    residual_final_metrics = _regression_metrics(test[FINAL_TARGET], reconstructed_final, test.sample_weight, test.canonical_project_id)
    residual_target_metrics = _regression_metrics(test[RESIDUAL_TARGET], residual_remaining, test.sample_weight, test.canonical_project_id)

    direct_mae = direct_metrics.get("MAE")
    residual_mae = residual_final_metrics.get("MAE")
    improvement = None
    if direct_mae not in (None, 0) and residual_mae is not None:
        improvement = round((float(direct_mae) - float(residual_mae)) / float(direct_mae) * 100, 3)

    rows = test[[
        "canonical_project_id", "project_name", "snapshot_date", "completion_year", "lifecycle_stage",
        CURRENT_OVERRUN, FINAL_TARGET, RESIDUAL_TARGET, "sample_weight",
    ]].copy()
    rows["direct_predicted_final_overrun"] = direct_final
    rows["residual_predicted_remaining_overrun"] = residual_remaining
    rows["residual_reconstructed_final_overrun"] = reconstructed_final
    rows["direct_error"] = rows.direct_predicted_final_overrun - rows[FINAL_TARGET]
    rows["residual_final_error"] = rows.residual_reconstructed_final_overrun - rows[FINAL_TARGET]

    train_digest = _cohort_digest(train)
    test_digest = _cohort_digest(test)
    report = {
        "experiment": "experiment_3_residual_remaining_overrun",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_period": [int(training_start), int(training_end)],
        "testing_period": [int(training_end) + 1, int(test_end)],
        "target_definition": {
            "current_observed_overrun": CURRENT_OVERRUN,
            "direct_target": FINAL_TARGET,
            "residual_target": f"{FINAL_TARGET} - {CURRENT_OVERRUN}",
            "final_reconstruction": f"{CURRENT_OVERRUN} + predicted_{RESIDUAL_TARGET}",
        },
        "comparison_control": {
            "same_train_snapshots": True,
            "same_test_snapshots": True,
            "same_features": True,
            "same_regressor_family": True,
            "train_snapshot_digest": train_digest,
            "test_snapshot_digest": test_digest,
            "training_rows": int(len(train)),
            "training_projects": int(train.canonical_project_id.nunique()),
            "test_rows": int(len(test)),
            "test_projects": int(test.canonical_project_id.nunique()),
        },
        "features_used": features,
        "feature_count": len(features),
        "selected_algorithm": algorithm,
        "direct_internal_algorithm_comparison": direct_internal,
        "direct_final_overrun_metrics": direct_metrics,
        "residual_reconstructed_final_overrun_metrics": residual_final_metrics,
        "residual_target_metrics": residual_target_metrics,
        "final_mae_improvement_percentage": improvement,
        "success_threshold": {
            "metric": "final-overrun MAE reduction",
            "minimum_percentage": 10.0,
            "passed": improvement is not None and improvement >= 10.0,
        },
        "direct_lifecycle_stage_metrics": _stage_cost_metrics(rows, "direct_predicted_final_overrun"),
        "residual_lifecycle_stage_metrics": _stage_cost_metrics(rows, "residual_reconstructed_final_overrun"),
        "interpretation": (
            "Residual forecasting is accepted only if reconstructed FINAL-overrun error improves on the direct model; "
            "a smaller residual-target error by itself is not evidence of better final forecasting."
        ),
    }

    if persist:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"residual_overrun_{int(training_start)}_{int(training_end)}"
        (REPORT_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        rows.to_csv(REPORT_DIR / f"{stem}_predictions.csv", index=False, date_format="%Y-%m-%d")
    return report
