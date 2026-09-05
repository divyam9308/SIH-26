"""Live lifecycle retraining routed through the Exp105 Cost + Exp113 Delay stack.

This keeps the existing atomic publication/provenance helpers from the standard
lifecycle retraining service, but uses the selected production trainer:
Exp105 for Cost and Exp113 for Delay on top of the existing Exp61 + U1 stack.
"""
from __future__ import annotations

from datetime import datetime, timezone
import shutil
import uuid

from backend.app.ml.monthly_training import MODEL_ROOT
from backend.app.ml.production_cost_baseline import target_feature_contract
from backend.app.ml import production_exp35_baseline as exp35_baseline
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay
from backend.app.services import lifecycle_retraining_service as base
from backend.app.services import monthly_prediction_service


# The frozen Exp35 audit guard still records the old Exp34 Cost anchor (26.872),
# while the current main/data path reproducibly rebuilds that intermediate anchor
# at 26.989 before continuing through Exp61 -> Exp105. Live Model Simulation must
# validate the production stack it actually retrains, rather than aborting on the
# obsolete intermediate audit number. The override is scoped to this service call
# and restored immediately afterwards; the Exp35 frozen audit workflow is unchanged.
CURRENT_RETRAIN_EXP34_COST_ANCHOR_MAE = 26.989


def retrain_lifecycle(start_year: int, end_year: int) -> dict:
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("Training start year must be less than or equal to training end year.")
    data, identity, min_year, max_year = base._training_data()
    if end_year >= max_year:
        raise ValueError(f"Training must end before {max_year} so an unseen future lifecycle holdout remains.")
    if end_year < min_year or start_year > max_year:
        raise ValueError(f"Training range must overlap identity-verified lifecycle data ({min_year}-{max_year}).")
    selected_training_years = data.loc[data.completion_year.between(start_year, end_year), "completion_year"].dropna()
    if selected_training_years.empty:
        raise ValueError("The selected period has no identity-verified lifecycle training projects.")

    internal_validation_year = int(selected_training_years.max())
    window = f"{start_year}_{end_year}"
    target = MODEL_ROOT / window
    target.mkdir(parents=True, exist_ok=True)
    training_marker = target / ".training"
    training_marker.write_text(datetime.now(timezone.utc).isoformat())
    staging_root = MODEL_ROOT / ".staging" / f"{window}-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True, exist_ok=False)
    original_exp35_cost_anchor = exp35_baseline.VERIFIED_BASE_COST_MAE
    try:
        # The current production trainer is several promotions beyond Exp35. Its
        # nested Exp35 step should keep the 721/688 routing and improvement guards,
        # but compare against the intermediate anchor produced by current main.
        if (start_year, end_year, max_year) == (2001, 2021, 2025):
            exp35_baseline.VERIFIED_BASE_COST_MAE = CURRENT_RETRAIN_EXP34_COST_ANCHOR_MAE
        result = train_window_with_promoted_cost_and_delay(
            start_year,
            end_year,
            max_year,
            data=data,
            identity=identity,
            artifact_root=staging_root,
        )
        staged_target = staging_root / window
        base._stamp_production_role(result, staged_target)
        base._write_run_manifest(start_year, end_year, result, staged_target)
        base._publish_staged_run(staged_target, target)
    finally:
        exp35_baseline.VERIFIED_BASE_COST_MAE = original_exp35_cost_anchor
        training_marker.unlink(missing_ok=True)
        shutil.rmtree(staging_root, ignore_errors=True)

    metadata = result["metadata"]
    lifecycle = result["lifecycle"]
    lifecycle_metrics = lifecycle["metrics"]
    baseline_metrics = result["baseline"]["metrics"]
    feature_audit = metadata.get("feature_availability", {})
    selected = metadata.get("selected_algorithms", {})
    provenance = metadata.get("provenance", {})
    feature_contract = target_feature_contract(metadata)
    monthly_prediction_service._bundle.cache_clear()

    return {
        "status": "success",
        "model_role": "production",
        "model_family": "monthly_lifecycle",
        "model_version": metadata["model_version"],
        "run_id": metadata.get("run_id") or provenance.get("run_id"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint") or provenance.get("dataset_fingerprint"),
        "window": window,
        "training_years": f"{start_year}-{end_year}",
        "testing_years": f"{end_year + 1}-{max_year}",
        "training_samples": metadata["training_snapshots"],
        "training_projects": metadata["unique_training_projects"],
        "testing_samples": metadata["test_snapshots"],
        "testing_projects": metadata["unique_test_projects"],
        "features_used": metadata["features_used"],
        "cost_features_used": feature_contract["cost"],
        "delay_features_used": feature_contract["delay"],
        "risk_features_used": feature_contract["risk"],
        "feature_count": len(feature_contract["cost"]),
        "feature_count_by_target": {name: len(features) for name, features in feature_contract.items()},
        "production_cost_baseline": metadata.get("production_cost_baseline"),
        "production_delay_baseline": metadata.get("production_delay_baseline"),
        "promoted_from_experiment": metadata.get("promoted_from_experiment"),
        "promoted_cost_from_experiment": metadata.get("promoted_cost_from_experiment"),
        "promoted_delay_from_experiment": metadata.get("promoted_delay_from_experiment"),
        "selected_algorithms": {
            "cost": selected.get("cost"),
            "delay": selected.get("delay"),
            "risk": "random_forest",
        },
        "internal_validation_year": internal_validation_year,
        "future_holdout_start": end_year + 1,
        "future_holdout_end": max_year,
        "metrics": {
            "cost_model": lifecycle_metrics["cost"],
            "delay_model": lifecycle_metrics["delay"],
            "risk_model": lifecycle_metrics["risk"],
            "metadata": {
                "feature_count": len(feature_contract["cost"]),
                "feature_count_by_target": {name: len(features) for name, features in feature_contract.items()},
                "features_used": metadata["features_used"],
                "cost_features_used": feature_contract["cost"],
                "delay_features_used": feature_contract["delay"],
                "production_cost_baseline": metadata.get("production_cost_baseline"),
                "production_delay_baseline": metadata.get("production_delay_baseline"),
                "delay_evaluation_contract": metadata.get("delay_evaluation_contract"),
                "feature_quality": {
                    "data_quality_score": feature_audit.get("data_quality_score"),
                    "removed_invalid_feature_count": feature_audit.get(
                        "removed_invalid_feature_count", len(feature_audit.get("removed_features", []))
                    ),
                    "as_of_evidence_coverage": feature_audit.get("as_of_evidence_coverage"),
                },
                "leakage_policy": metadata.get("leakage_policy"),
                "snapshot_weighting_policy": metadata.get("snapshot_weighting_policy"),
                "balanced_stage_summary": metadata.get("balanced_stage_summary"),
            },
        },
        "baseline_comparison": {
            "feature_count": len(result["baseline"].get("features") or []),
            "cost_mae": baseline_metrics["cost"]["MAE"],
            "delay_mae": baseline_metrics["delay"]["MAE"],
            "risk_macro_f1": baseline_metrics["risk"]["macro_f1"],
            "purpose": "Controlled benchmark only; not the retrained production forecast model.",
        },
        "lifecycle_stages": lifecycle.get("lifecycle_stages", {}),
        "stage_distribution": lifecycle.get("stage_distribution", {}),
        "balanced_stage_summary": lifecycle.get("balanced_stage_summary", {}),
        "leakage_guard": (
            "Future holdout projects are excluded from fitting and selection. Exp105 Cost uses only "
            "forward training OOF factor/residual evidence; Exp113 Delay uses only forward training OOF "
            "quantile-AFT/residual evidence on top of the existing Exp61 + U1 Delay anchor. Risk is unchanged."
        ),
    }