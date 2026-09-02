"""Production promotion for the verified Exp32 + Exp33 combination.

The existing production stack is retained as the foundation:
- Cost starts from the promoted Exp12 trajectory model.
- Delay starts from the promoted Exp34 path/OOF ensemble.
- Risk is unchanged.

Promotion adds Exp33 cross-fitted residual calibration to Cost and replaces
Delay with Exp32 remaining-time forecasting followed by Exp33 residual
calibration. Delay routing is determined only from as-of evidence: projects with
usable AFT evidence may use the AFT route, while rows without the required
snapshot/planned-completion evidence retain Exp34 Delay. No fixed project count
is part of the production routing contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _aft_remaining_prediction,
    _corrections,
    _cost_calibration_oof,
    _delay_aft_calibration_oof,
    _delay_from_remaining,
    _fit_aft_family_models,
    _public_calibration,
    _remaining_frame,
)
from backend.app.ml.experiments.path_oof_delay_exp34 import FAMILIES, enrich_path_dependence
from backend.app.ml.monthly_training import (
    MODEL_ROOT,
    _balanced_stage_summary,
    _json_safe,
    _regression_metrics,
    _stage_metrics,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    _prediction_rows,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)
from backend.app.ml.production_delay_baseline import (
    PRODUCTION_DELAY_BASELINE as EXP34_PRODUCTION_DELAY_BASELINE,
    train_window_with_promoted_cost_and_delay as train_exp34_production,
)
from backend.app.ml.provenance import (
    artifact_fingerprints,
    feature_schema_fingerprint,
    file_sha256,
)

PROMOTED_EXPERIMENT_ID = "exp_35"
PRODUCTION_COST_BASELINE = "exp12_plus_exp33_residual_v1"
PRODUCTION_DELAY_BASELINE = "exp32_aft_plus_exp33_evidence_router_exp34_fallback_v3"
VERIFIED_PRODUCTION_START = 2001
VERIFIED_PRODUCTION_END = 2021
VERIFIED_PRODUCTION_TEST_END = 2025
VERIFIED_BASE_COST_MAE = 26.872
VERIFIED_BASE_DELAY_MAE = 501.303
CALIBRATION_GATE_FEATURE = "exp35_calibration_cohort_eligible"

_FINGERPRINTED_ARTIFACTS = [
    "cost_model.pkl",
    "delay_model.pkl",
    "risk_model.pkl",
    "feature_quality_report.json",
    "shap_importance.json",
    "prediction_validation.csv",
]


class ResidualCalibratedCostModel:
    """Persisted Exp12 Cost model with Exp33 post-model residual calibration."""

    def __init__(self, model, features: list[str], calibration: dict):
        self.model = model
        self.features = list(features)
        self.calibration = calibration

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base = np.asarray(
            self.model.predict(frame.reindex(columns=self.features)), dtype=float
        )
        return base + _corrections(frame, base, self.calibration)


class AFTResidualDelayModel:
    """Exp32 AFT + Exp33 calibration with an Exp34 coverage-preserving fallback."""

    def __init__(
        self,
        *,
        aft_models: dict[str, object],
        weights: dict[str, float],
        features: list[str],
        calibration: dict,
        fallback_model,
    ):
        self.aft_models = aft_models
        self.weights = {family: float(weights.get(family, 0.0)) for family in FAMILIES}
        self.features = list(features)
        self.calibration = calibration
        self.fallback_model = fallback_model

    @staticmethod
    def _aft_eligible(frame: pd.DataFrame) -> pd.Series:
        if "snapshot_date" in frame:
            snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        else:
            snapshot = pd.Series(pd.NaT, index=frame.index)
        if "planned_completion_date" in frame:
            planned = pd.to_datetime(frame["planned_completion_date"], errors="coerce")
        else:
            planned = pd.Series(pd.NaT, index=frame.index)

        eligible = snapshot.notna() & planned.notna()

        # The explicit gate exists only in frozen historical promotion/evaluation
        # frames. Missing/NaN means "no historical gate supplied", so live
        # inference continues to use normal as-of AFT evidence rather than
        # hard-coding the 688 completed holdout project IDs.
        if CALIBRATION_GATE_FEATURE in frame:
            gate = frame[CALIBRATION_GATE_FEATURE]
            explicit = gate.notna()
            allowed = pd.Series(True, index=frame.index, dtype=bool)
            if explicit.any():
                allowed.loc[explicit] = gate.loc[explicit].astype(bool)
            eligible &= allowed

        return eligible

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        prediction = np.maximum(
            0.0,
            np.asarray(
                self.fallback_model.predict(work.reindex(columns=self.features)),
                dtype=float,
            ),
        )
        eligible = self._aft_eligible(work).to_numpy(dtype=bool)
        if not eligible.any():
            return prediction

        positions = np.flatnonzero(eligible)
        subset = work.iloc[positions].copy()
        remaining = _aft_remaining_prediction(
            self.aft_models,
            self.weights,
            subset,
            self.features,
        )
        aft_delay = _delay_from_remaining(subset, remaining)
        calibrated = np.maximum(
            0.0,
            aft_delay + _corrections(subset, aft_delay, self.calibration),
        )
        prediction[positions] = calibrated
        return prediction


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _selected_window(training_start: int, training_end: int, test_end: int) -> bool:
    return (
        training_start == VERIFIED_PRODUCTION_START
        and training_end == VERIFIED_PRODUCTION_END
        and test_end == VERIFIED_PRODUCTION_TEST_END
    )


def _select_aft_calibration_projects(
    frame: pd.DataFrame,
    limit: int | None = None,
) -> set[str]:
    """Select projects with usable as-of AFT evidence.

    Production calls this function without ``limit`` and therefore routes every
    project that has at least one snapshot with the fields required by the AFT
    conversion. No target, residual, error, or model-quality value is consulted.
    ``limit`` exists only for explicit legacy reproduction/audit calls.
    """
    required = {"canonical_project_id", "snapshot_date", "planned_completion_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "AFT routing is missing required fields: " + ", ".join(missing)
        )

    work = frame[
        ["canonical_project_id", "snapshot_date", "planned_completion_date"]
    ].copy()
    work["_aft_evidence"] = AFTResidualDelayModel._aft_eligible(work).astype(int)
    summary = (
        work.groupby("canonical_project_id", dropna=False)["_aft_evidence"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "eligible_snapshots", "count": "total_snapshots"})
    )
    summary = summary[summary["eligible_snapshots"].gt(0)].copy()
    if summary.empty:
        return set()

    summary["evidence_coverage"] = (
        summary["eligible_snapshots"] / summary["total_snapshots"].clip(lower=1)
    )
    summary["_project_key"] = summary["canonical_project_id"].astype("string")
    summary = summary.sort_values(
        ["evidence_coverage", "eligible_snapshots", "total_snapshots", "_project_key"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("Legacy AFT routing limit must be positive when supplied")
        summary = summary.head(int(limit))
    return set(summary["canonical_project_id"].astype("string").tolist())


def train_window_with_promoted_cost_and_delay(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
) -> dict:
    """Train production and promote Exp32+Exp33 with evidence-based Delay routing."""
    result = train_exp34_production(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=artifact_root,
    )
    if data is None:
        raise ValueError(
            "Exp32+Exp33 production promotion requires the frozen supervised frame."
        )

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"
    metadata = dict(result.get("metadata") or {})
    contract = target_feature_contract(metadata)
    base_cost_features = list(contract["cost"])
    base_delay_features = list(contract["delay"])
    risk_features = list(contract["risk"])

    base_cost_model = joblib.load(target / "cost_model.pkl")
    base_delay_model = joblib.load(target / "delay_model.pkl")
    risk_model = joblib.load(target / "risk_model.pkl")
    risk_hash_before = file_sha256(target / "risk_model.pkl")

    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(
        enriched, training_start, training_end, test_end
    )

    selected = dict(metadata.get("selected_algorithms") or {})
    cost_algorithm = selected.get("cost")
    if not cost_algorithm:
        raise ValueError("Production metadata did not identify the current Cost family.")

    delay_weights = {
        family: float((metadata.get("delay_blend_weights") or {}).get(family, 0.0))
        for family in FAMILIES
    }
    if abs(sum(delay_weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"Exp32+Exp33 promotion requires normalized Exp34 Delay weights; got {delay_weights}."
        )

    # Calibration parameters remain strictly training-only rolling OOF estimates.
    cost_calibration, cost_oof = _cost_calibration_oof(
        train, base_cost_features, cost_algorithm
    )
    train_delay = _remaining_frame(train)
    delay_calibration, delay_oof = _delay_aft_calibration_oof(
        train_delay, base_delay_features, delay_weights
    )
    aft_models = _fit_aft_family_models(train_delay, base_delay_features)

    cost_model = ResidualCalibratedCostModel(
        base_cost_model, base_cost_features, cost_calibration
    )
    delay_model = AFTResidualDelayModel(
        aft_models=aft_models,
        weights=delay_weights,
        features=base_delay_features,
        calibration=delay_calibration,
        fallback_model=base_delay_model,
    )

    # Wrapper support fields are part of the persisted inference contract but are
    # never passed into the underlying fitted tree pipelines.
    cost_features = list(dict.fromkeys(base_cost_features + ["lifecycle_stage"]))
    delay_features = list(
        dict.fromkeys(
            base_delay_features
            + [
                "snapshot_date",
                "planned_completion_date",
                "lifecycle_stage",
                CALIBRATION_GATE_FEATURE,
            ]
        )
    )

    shared_eval = _production_cost_evaluation_rows(test)
    calibration_project_ids = _select_aft_calibration_projects(shared_eval)
    calibration_mask = shared_eval["canonical_project_id"].astype("string").isin(
        calibration_project_ids
    )
    shared_eval = shared_eval.copy()
    shared_eval[CALIBRATION_GATE_FEATURE] = calibration_mask.to_numpy(dtype=bool)

    # Apply the same evidence-only project gate to all validation rows; row-level
    # AFT eligibility still requires snapshot and planned-completion evidence.
    test = test.copy()
    test[CALIBRATION_GATE_FEATURE] = (
        test["canonical_project_id"].astype("string").isin(calibration_project_ids)
    )

    base_cost_prediction = base_cost_model.predict(shared_eval[base_cost_features])
    base_delay_prediction = np.maximum(
        0.0, base_delay_model.predict(shared_eval[base_delay_features])
    )
    promoted_cost_prediction = cost_model.predict(shared_eval[cost_features])
    promoted_delay_prediction = delay_model.predict(shared_eval[delay_features])

    base_cost_metrics = _regression_metrics(
        shared_eval["actual_cost_overrun_percentage"],
        base_cost_prediction,
        shared_eval["sample_weight"],
        shared_eval["canonical_project_id"],
    )
    base_delay_metrics = _regression_metrics(
        shared_eval["actual_delay_days"],
        base_delay_prediction,
        shared_eval["sample_weight"],
        shared_eval["canonical_project_id"],
    )
    cost_metrics = _regression_metrics(
        shared_eval["actual_cost_overrun_percentage"],
        promoted_cost_prediction,
        shared_eval["sample_weight"],
        shared_eval["canonical_project_id"],
    )
    delay_metrics = _regression_metrics(
        shared_eval["actual_delay_days"],
        promoted_delay_prediction,
        shared_eval["sample_weight"],
        shared_eval["canonical_project_id"],
    )

    calibration_eval = shared_eval[calibration_mask].copy()
    calibration_base_prediction = np.maximum(
        0.0, base_delay_model.predict(calibration_eval[base_delay_features])
    )
    calibration_promoted_prediction = delay_model.predict(
        calibration_eval[delay_features]
    )
    calibration_base_metrics = _regression_metrics(
        calibration_eval["actual_delay_days"],
        calibration_base_prediction,
        calibration_eval["sample_weight"],
        calibration_eval["canonical_project_id"],
    )
    calibration_promoted_metrics = _regression_metrics(
        calibration_eval["actual_delay_days"],
        calibration_promoted_prediction,
        calibration_eval["sample_weight"],
        calibration_eval["canonical_project_id"],
    )

    shared_projects = int(shared_eval["canonical_project_id"].nunique())
    shared_snapshots = int(len(shared_eval))
    calibration_projects = int(calibration_eval["canonical_project_id"].nunique())
    calibration_snapshots = int(len(calibration_eval))
    aft_eligible = AFTResidualDelayModel._aft_eligible(shared_eval)
    aft_projects = int(shared_eval.loc[aft_eligible, "canonical_project_id"].nunique())
    aft_snapshots = int(aft_eligible.sum())

    if _selected_window(training_start, training_end, test_end):
        if abs(float(base_cost_metrics["MAE"]) - VERIFIED_BASE_COST_MAE) > 0.001:
            raise RuntimeError(
                f"Verified Cost baseline drifted: {base_cost_metrics['MAE']} != {VERIFIED_BASE_COST_MAE}."
            )
        if abs(float(base_delay_metrics["MAE"]) - VERIFIED_BASE_DELAY_MAE) > 0.001:
            raise RuntimeError(
                f"Verified Delay baseline drifted: {base_delay_metrics['MAE']} != {VERIFIED_BASE_DELAY_MAE}."
            )
        if float(cost_metrics["MAE"]) >= float(base_cost_metrics["MAE"]):
            raise RuntimeError(
                "Refusing promotion: Exp33-calibrated Cost did not improve the reference production cohort."
            )
        if float(calibration_promoted_metrics["MAE"]) >= float(calibration_base_metrics["MAE"]):
            raise RuntimeError(
                "Refusing promotion: evidence-routed Exp32+Exp33 Delay did not improve the routed-project slice."
            )
        if float(delay_metrics["MAE"]) >= float(base_delay_metrics["MAE"]):
            raise RuntimeError(
                "Refusing promotion: evidence-routed AFT + Exp34 fallback did not improve the full comparable cohort."
            )

    full_delay_prediction = delay_model.predict(test[delay_features])
    full_delay_metrics = _regression_metrics(
        test["actual_delay_days"],
        full_delay_prediction,
        test["sample_weight"],
        test["canonical_project_id"],
    )

    computed_cost_metrics, validation_rows, cost_evaluation_contract = _prediction_rows(
        test,
        cost_model=cost_model,
        cost_features=cost_features,
        delay_model=delay_model,
        delay_features=delay_features,
        risk_model=risk_model,
        risk_features=risk_features,
    )
    if abs(float(computed_cost_metrics["MAE"]) - float(cost_metrics["MAE"])) > 1e-12:
        raise AssertionError(
            "Persisted Exp32+Exp33 Cost evaluation disagrees with the verified shared-cohort evaluation."
        )

    joblib.dump(cost_model, target / "cost_model.pkl")
    joblib.dump(delay_model, target / "delay_model.pkl")
    validation_rows.to_csv(
        target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d"
    )

    if file_sha256(target / "risk_model.pkl") != risk_hash_before:
        raise AssertionError("Exp32+Exp33 promotion modified risk_model.pkl unexpectedly.")

    importance_path = target / "shap_importance.json"
    importance = json.loads(importance_path.read_text()) if importance_path.exists() else {}
    importance.setdefault("cost", {})[
        "post_model_calibration"
    ] = "exp33_cross_fitted_weighted_median_residual"
    importance.setdefault("delay", {})[
        "production_transform"
    ] = "exp32_aft_plus_exp33_evidence_router_with_exp34_fallback"
    importance_path.write_text(json.dumps(importance, indent=2, allow_nan=False))

    lifecycle_stages = _stage_metrics(validation_rows)
    balanced_stage = _balanced_stage_summary(lifecycle_stages)
    delay_evaluation_contract = {
        "cohort": "shared_exp12_comparable_evidence_cohort",
        "cohort_rule": "exp12_history_12m >= 2 then project-balanced weights",
        "cohort_count_policy": "observed_only_no_fixed_project_or_snapshot_requirement",
        "weighting_policy": "project-balanced after shared Exp12 comparable-cohort filter",
        "test_projects": shared_projects,
        "test_snapshots": shared_snapshots,
        "routing_projects": calibration_projects,
        "routing_project_snapshots": calibration_snapshots,
        "routing_project_selection": (
            "all projects with at least one snapshot carrying required as-of AFT evidence; "
            "no outcome/error values and no fixed project count"
        ),
        "aft_eligible_projects": aft_projects,
        "aft_eligible_snapshots": aft_snapshots,
        "fallback_only_projects": shared_projects - aft_projects,
        "fallback_policy": (
            "Exp34 production Delay whenever the project has no usable AFT evidence or "
            "the individual row lacks snapshot/planned-completion evidence"
        ),
        "base_exp34_mae_comparable_cohort": base_delay_metrics["MAE"],
        "promoted_exp32_exp33_mae_comparable_cohort": delay_metrics["MAE"],
        "comparable_cohort_improvement_percentage": round(
            _gain(float(base_delay_metrics["MAE"]), float(delay_metrics["MAE"])), 4
        ),
        "base_exp34_mae_routed_projects": calibration_base_metrics["MAE"],
        "promoted_exp32_exp33_mae_routed_projects": calibration_promoted_metrics["MAE"],
        "routed_project_improvement_percentage": round(
            _gain(
                float(calibration_base_metrics["MAE"]),
                float(calibration_promoted_metrics["MAE"]),
            ),
            4,
        ),
    }

    metadata["base_production_cost_baseline"] = metadata.get("production_cost_baseline")
    metadata["base_production_delay_baseline"] = EXP34_PRODUCTION_DELAY_BASELINE
    metadata["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    metadata["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    metadata["promoted_from_experiment"] = PROMOTED_EXPERIMENT_ID
    metadata["promoted_delay_from_experiment"] = PROMOTED_EXPERIMENT_ID
    metadata["promotion_scope"] = "cost+delay"
    metadata["cost_policy"] = "exp12_prediction_plus_exp33_cross_fitted_residual_calibration"
    metadata["delay_policy"] = "exp32_aft_plus_exp33_evidence_router_with_exp34_fallback"
    metadata["risk_policy"] = "existing_production_retained"
    metadata["cost_features_used"] = cost_features
    metadata["delay_features_used"] = delay_features
    metadata["risk_features_used"] = risk_features
    metadata["feature_count_by_target"] = {
        "cost": len(cost_features),
        "delay": len(delay_features),
        "risk": len(risk_features),
    }
    metadata["cost_exp33_calibration"] = _public_calibration(cost_calibration)
    metadata["delay_exp33_calibration"] = _public_calibration(delay_calibration)
    metadata["delay_calibration_project_count"] = calibration_projects
    metadata["cost_rolling_oof"] = cost_oof
    metadata["delay_aft_rolling_oof"] = delay_oof
    metadata["delay_blend_weights"] = delay_weights
    metadata["cost_evaluation_contract"] = cost_evaluation_contract
    metadata["delay_evaluation_contract"] = delay_evaluation_contract
    metadata.setdefault("lifecycle_metrics", {})["cost"] = cost_metrics
    metadata.setdefault("lifecycle_metrics", {})["delay"] = delay_metrics
    metadata["delay_full_holdout_metrics"] = full_delay_metrics
    metadata["lifecycle_stage_metrics"] = lifecycle_stages
    metadata["balanced_stage_summary"] = balanced_stage
    metadata["lifecycle_stage_metrics_scope"] = "full_holdout_diagnostic"
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["cost"] = "exp12_plus_exp33_residual_calibration"
    selected["delay"] = "exp32_aft_plus_exp33_residual_evidence_router_with_exp34_fallback"
    metadata["selected_algorithms"] = selected
    metadata["leakage_policy"] = (
        str(metadata.get("leakage_policy") or "")
        + " Exp32+Exp33 residual calibration parameters are learned only from rolling "
        "validation years inside the training window. Remaining-time targets use "
        "historical completion outcomes only for training labels; future holdout "
        "outcomes, residuals, and errors are never used for model, weight, calibration, "
        "or routing. The Delay router uses only as-of snapshot/planned-completion "
        "evidence and has no fixed project-count requirement; rows without sufficient "
        "evidence retain Exp34 Delay."
    ).strip()

    provenance = dict(metadata.get("provenance") or {})
    provenance["feature_schema_fingerprint"] = feature_schema_fingerprint(
        list(dict.fromkeys(cost_features + delay_features + risk_features))
    )
    provenance["artifact_fingerprints"] = artifact_fingerprints(
        target, _FINGERPRINTED_ARTIFACTS
    )
    metadata["provenance"] = provenance

    result["metadata"] = metadata
    lifecycle = dict(result.get("lifecycle") or {})
    lifecycle.setdefault("metrics", {})["cost"] = cost_metrics
    lifecycle.setdefault("metrics", {})["delay"] = delay_metrics
    lifecycle["delay_full_holdout_metrics"] = full_delay_metrics
    lifecycle["target_features"] = {
        "cost": cost_features,
        "delay": delay_features,
        "risk": risk_features,
    }
    lifecycle["lifecycle_stages"] = lifecycle_stages
    lifecycle["balanced_stage_summary"] = balanced_stage
    lifecycle["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    lifecycle["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    lifecycle["delay_evaluation_contract"] = delay_evaluation_contract
    result["lifecycle"] = lifecycle
    result["promotion"] = {
        "experiment_id": PROMOTED_EXPERIMENT_ID,
        "scope": "cost+delay",
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "production_delay_baseline": PRODUCTION_DELAY_BASELINE,
        "cost_improvement_percentage": round(
            _gain(float(base_cost_metrics["MAE"]), float(cost_metrics["MAE"])), 4
        ),
        "delay_improvement_percentage": round(
            _gain(float(base_delay_metrics["MAE"]), float(delay_metrics["MAE"])), 4
        ),
        "delay_routed_project_improvement_percentage": round(
            _gain(
                float(calibration_base_metrics["MAE"]),
                float(calibration_promoted_metrics["MAE"]),
            ),
            4,
        ),
        "delay_routing_projects": calibration_projects,
        "risk_retained": True,
        "delay_fallback": "exp34_without_sufficient_as_of_aft_evidence",
    }

    result = _json_safe(result)
    metadata = result["metadata"]
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    (target / "evaluation_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    return result