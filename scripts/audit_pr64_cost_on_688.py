from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from backend.app.ml.experiments.exp35_aft_residual_combo import _cost_calibration_oof
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
    train_window_with_promoted_cost,
)
from backend.app.ml.production_exp35_baseline import (
    ResidualCalibratedCostModel,
    VERIFIED_AFT_CALIBRATION_PROJECTS,
    VERIFIED_BASE_COST_MAE,
    VERIFIED_PRODUCTION_PROJECTS,
    VERIFIED_PRODUCTION_SNAPSHOTS,
    _select_aft_calibration_projects,
)
from backend.app.services.lifecycle_retraining_service import _training_data


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _cost_metrics(frame: pd.DataFrame, prediction) -> dict:
    return _regression_metrics(
        frame["actual_cost_overrun_percentage"],
        prediction,
        frame["sample_weight"],
        frame["canonical_project_id"],
    )


def run_audit(*, output: Path) -> dict:
    training_start, training_end, test_end = 2001, 2021, 2025
    data, identity, _, _ = _training_data()

    # Train the exact Exp12 production Cost path from scratch in an isolated temp
    # artifact root. Nothing here writes to the repository's production models.
    with tempfile.TemporaryDirectory(prefix="pr64-cost-688-") as tmp:
        artifact_root = Path(tmp)
        base_result = train_window_with_promoted_cost(
            training_start,
            training_end,
            test_end,
            data=data,
            identity=identity,
            artifact_root=artifact_root,
        )
        target = artifact_root / f"{training_start}_{training_end}"
        base_cost_model = joblib.load(target / "cost_model.pkl")

        metadata = dict(base_result.get("metadata") or {})
        contract = target_feature_contract(metadata)
        cost_features = list(contract["cost"])
        cost_algorithm = (metadata.get("selected_algorithms") or {}).get("cost")
        if not cost_algorithm:
            raise RuntimeError("Fresh production run did not identify a Cost algorithm.")

        # Match PR64's Exp35 preprocessing and training-only OOF Cost calibration.
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
        cost_calibration, _ = _cost_calibration_oof(
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

        # Select exactly the same evidence-only 688 project IDs used by PR64's
        # frozen Delay audit, then recalculate project-balanced weights after the
        # subset filter before scoring Cost.
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
            "training_window": [training_start, training_end],
            "test_end": test_end,
            "selection_policy": (
                "exact PR64 evidence-only AFT calibration cohort selector; "
                "no target, residual, error, or model-quality values used"
            ),
            "future_holdout_used_to_fit_cost_calibration": False,
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
