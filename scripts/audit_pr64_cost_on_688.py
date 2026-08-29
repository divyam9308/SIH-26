from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.exp35_aft_residual_combo import _cost_calibration_oof
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.experiments.trajectory_exp12_v2 import (
    _select_target_features,
    _usable_features,
)
from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    assign_project_balanced_weights,
    build_training_dataset,
)
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    _select_regressor,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
)
from backend.app.ml.production_exp35_baseline import (
    ResidualCalibratedCostModel,
    VERIFIED_AFT_CALIBRATION_PROJECTS,
    VERIFIED_BASE_COST_MAE,
    VERIFIED_PRODUCTION_PROJECTS,
    VERIFIED_PRODUCTION_SNAPSHOTS,
    _select_aft_calibration_projects,
)


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _cost_metrics(frame: pd.DataFrame, prediction) -> dict:
    return _regression_metrics(
        frame["actual_cost_overrun_percentage"],
        prediction,
        frame["sample_weight"],
        frame["canonical_project_id"],
    )


def _fresh_exact_exp12_cost(data: pd.DataFrame, start: int, end: int, test_end: int):
    """Freshly reproduce current production Cost without fitting unrelated targets.

    This is the Cost-only equivalent of scripts/run_fast_current_experiment.py:
    same leakage audit, feature contract, model-family selection, Exp12 trajectory
    selection, seeds and final fit; Delay/Risk/SHAP/ablations are skipped because
    they cannot influence Cost.
    """
    raw_train, _ = temporal_project_split(data, start, end, test_end)
    audit = audit_features(
        raw_train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage same-snapshot signal; evaluated by production ablation",
            "cost_escalation_percentage": "same-snapshot revised-cost derivative; evaluated by production ablation",
        },
    )
    production_features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    cost_algorithm, _discarded, algorithm_comparisons = _select_regressor(
        raw_train,
        production_features,
        "actual_cost_overrun_percentage",
        26203,
    )

    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(
        enriched["completion_year"], errors="coerce"
    )
    enriched_train, _ = temporal_project_split(enriched, start, end, test_end)
    usable, trajectory_audit = _usable_features(enriched_train)
    cost_added, cost_group, trajectory_comparisons = _select_target_features(
        enriched_train,
        production_features,
        usable,
        "actual_cost_overrun_percentage",
        cost_algorithm,
        PRODUCTION_COST_SEED,
    )
    cost_features = list(dict.fromkeys(production_features + cost_added))
    model = _fit_pipeline(
        _regressors(PRODUCTION_COST_SEED)[cost_algorithm],
        enriched_train,
        cost_features,
        "actual_cost_overrun_percentage",
    )
    diagnostics = {
        "cost_algorithm": cost_algorithm,
        "production_features": production_features,
        "cost_features": cost_features,
        "cost_trajectory_feature_group": cost_group,
        "cost_trajectory_features": cost_added,
        "algorithm_comparisons": algorithm_comparisons,
        "trajectory_feature_availability": trajectory_audit,
        "trajectory_comparisons": trajectory_comparisons,
    }
    return model, cost_features, cost_algorithm, diagnostics


def run_audit(*, output: Path) -> dict:
    training_start, training_end, test_end = 2001, 2021, 2025
    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")

    base_cost_model, cost_features, cost_algorithm, diagnostics = _fresh_exact_exp12_cost(
        data, training_start, training_end, test_end
    )

    # Match PR64's exact Exp35 preprocessing and training-only rolling OOF Cost
    # calibration. Path enrichment only adds fields; the Cost feature contract is
    # the freshly selected Exp12 production feature list above.
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(
        enriched["completion_year"], errors="coerce"
    )
    enriched["snapshot_date"] = pd.to_datetime(
        enriched["snapshot_date"], errors="coerce"
    )
    train, test = temporal_project_split(
        enriched, training_start, training_end, test_end
    )
    cost_calibration, cost_oof = _cost_calibration_oof(
        train, cost_features, cost_algorithm
    )
    promoted_cost_model = ResidualCalibratedCostModel(
        base_cost_model, cost_features, cost_calibration
    )

    full = _production_cost_evaluation_rows(test)
    full_base_prediction = base_cost_model.predict(full[cost_features])
    full_promoted_prediction = promoted_cost_model.predict(full)
    full_base = _cost_metrics(full, full_base_prediction)
    full_promoted = _cost_metrics(full, full_promoted_prediction)

    full_projects = int(full["canonical_project_id"].nunique())
    full_snapshots = int(len(full))
    if full_projects != VERIFIED_PRODUCTION_PROJECTS or full_snapshots != VERIFIED_PRODUCTION_SNAPSHOTS:
        raise RuntimeError(
            "Fresh production cohort mismatch: "
            f"expected {VERIFIED_PRODUCTION_PROJECTS}/{VERIFIED_PRODUCTION_SNAPSHOTS}, "
            f"found {full_projects}/{full_snapshots}."
        )
    if abs(float(full_base["MAE"]) - VERIFIED_BASE_COST_MAE) > 0.001:
        raise RuntimeError(
            "Fresh Exp12 Cost baseline did not reproduce PR64 production: "
            f"expected {VERIFIED_BASE_COST_MAE:.3f}, got {float(full_base['MAE']):.3f}."
        )
    if abs(float(full_promoted["MAE"]) - 26.287) > 0.001:
        raise RuntimeError(
            "Fresh Exp35 Cost path did not reproduce PR64 full-cohort result: "
            f"expected 26.287, got {float(full_promoted['MAE']):.3f}."
        )

    # Select exactly the same evidence-only 688 project IDs used by PR64's frozen
    # Delay audit. Target outcomes, residuals and errors are not used to select
    # the subset. Recalculate project-balanced weights after filtering.
    selected_ids = _select_aft_calibration_projects(
        full, limit=VERIFIED_AFT_CALIBRATION_PROJECTS
    )
    selected = full[
        full["canonical_project_id"].astype("string").isin(selected_ids)
    ].copy()
    selected = assign_project_balanced_weights(selected)

    selected_projects = int(selected["canonical_project_id"].nunique())
    if selected_projects != VERIFIED_AFT_CALIBRATION_PROJECTS:
        raise RuntimeError(
            f"Expected exactly {VERIFIED_AFT_CALIBRATION_PROJECTS} selected projects; "
            f"found {selected_projects}."
        )

    selected_base_prediction = base_cost_model.predict(selected[cost_features])
    selected_promoted_prediction = promoted_cost_model.predict(selected)
    selected_base = _cost_metrics(selected, selected_base_prediction)
    selected_promoted = _cost_metrics(selected, selected_promoted_prediction)

    excluded = full[
        ~full["canonical_project_id"].astype("string").isin(selected_ids)
    ].copy()
    excluded = assign_project_balanced_weights(excluded)
    excluded_base = _cost_metrics(
        excluded, base_cost_model.predict(excluded[cost_features])
    )
    excluded_promoted = _cost_metrics(
        excluded, promoted_cost_model.predict(excluded)
    )

    payload = {
        "audit": "PR64 Cost MAE on exact fixed 688-project evidence cohort",
        "audit_mode": (
            "fresh exact Cost-only production training; skips Delay/Risk/SHAP/ablations/publication"
        ),
        "training_window": [training_start, training_end],
        "test_end": test_end,
        "selection_policy": (
            "exact PR64 evidence-only AFT calibration cohort selector; "
            "no target, residual, error, or model-quality values used"
        ),
        "future_holdout_used_to_fit_cost_calibration": False,
        "fresh_cost_training": {
            "algorithm": cost_algorithm,
            "trajectory_feature_group": diagnostics["cost_trajectory_feature_group"],
            "cost_feature_count": len(cost_features),
            "cost_oof_rows": int(cost_calibration.get("oof_rows", 0)),
            "rolling_oof_folds": len(cost_oof),
        },
        "full_721": {
            "projects": full_projects,
            "snapshots": full_snapshots,
            "production_exp12_cost_mae": round(float(full_base["MAE"]), 6),
            "pr64_exp12_plus_exp33_cost_mae": round(float(full_promoted["MAE"]), 6),
            "improvement_percentage": round(
                _gain(float(full_base["MAE"]), float(full_promoted["MAE"])), 6
            ),
        },
        "selected_688": {
            "projects": selected_projects,
            "snapshots": int(len(selected)),
            "production_exp12_cost_mae": round(float(selected_base["MAE"]), 6),
            "pr64_exp12_plus_exp33_cost_mae": round(float(selected_promoted["MAE"]), 6),
            "improvement_percentage": round(
                _gain(float(selected_base["MAE"]), float(selected_promoted["MAE"])), 6
            ),
        },
        "excluded_33": {
            "projects": int(excluded["canonical_project_id"].nunique()),
            "snapshots": int(len(excluded)),
            "production_exp12_cost_mae": round(float(excluded_base["MAE"]), 6),
            "pr64_exp12_plus_exp33_cost_mae": round(float(excluded_promoted["MAE"]), 6),
            "improvement_percentage": round(
                _gain(float(excluded_base["MAE"]), float(excluded_promoted["MAE"])), 6
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print("PR64_COST_688_AUDIT=" + json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit_outputs/pr64_cost_688.json"),
    )
    args = parser.parse_args()
    run_audit(output=args.output)


if __name__ == "__main__":
    main()
