"""Production Delay baseline promoted from verified Experiment 34.

Cost remains the promoted Experiment 12 trajectory model. Risk remains the
existing production classifier. This module replaces only Delay with the
Experiment 34 path-dependence + rolling-OOF ensemble that improved both verified
future windows.

For the selected production window (2001-2021 -> 2022-2025), the official
headline evidence cohort is the same verified Exp12-comparable 721-project
cohort used for production Cost. Full-holdout Delay metrics are also retained as
a diagnostic so the 728-project Exp34 experiment result remains auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    FAMILIES,
    PATH_FEATURES,
    _fit_delay_family_models,
    _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import (
    MODEL_ROOT,
    _balanced_stage_summary,
    _importance,
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
    train_window_with_promoted_cost,
)
from backend.app.ml.provenance import (
    artifact_fingerprints,
    feature_schema_fingerprint,
    file_sha256,
)

PROMOTED_DELAY_EXPERIMENT_ID = "exp_34"
PRODUCTION_DELAY_BASELINE = "exp34_path_oof_ensemble_v1"
DEFAULT_PRODUCTION_WINDOW = "2001_2021"
VERIFIED_PRODUCTION_START = 2001
VERIFIED_PRODUCTION_END = 2021
VERIFIED_PRODUCTION_TEST_END = 2025
VERIFIED_PRODUCTION_EVIDENCE_PROJECTS = 721
PRODUCTION_DELAY_EVALUATION_COHORT = "shared_exp12_comparable_721_project_cohort"

_FINGERPRINTED_ARTIFACTS = [
    "cost_model.pkl",
    "delay_model.pkl",
    "risk_model.pkl",
    "feature_quality_report.json",
    "shap_importance.json",
    "prediction_validation.csv",
]


class ProductionDelayBlendModel:
    """Single persisted production model wrapping Exp34's three-family blend."""

    def __init__(self, models: dict, weights: dict[str, float], features: list[str]):
        self.models = models
        self.weights = {name: float(weights.get(name, 0.0)) for name in FAMILIES}
        self.features = list(features)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        data = frame.reindex(columns=self.features)
        prediction = np.zeros(len(data), dtype=float)
        for family in FAMILIES:
            prediction += self.weights[family] * self.models[family].predict(data)
        return prediction


def enrich_history_for_delay_production(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach causal Exp34 path features to a production inference history."""
    return enrich_path_dependence(frame, history=frame)


def _delay_importance(
    models: dict,
    weights: dict[str, float],
    train: pd.DataFrame,
    features: list[str],
) -> dict:
    """Blend family-level global importances with the fitted OOF weights."""
    sample = train.tail(min(500, len(train)))
    aggregate = {feature: 0.0 for feature in features}
    family_methods: dict[str, str] = {}
    for family in FAMILIES:
        weight = float(weights.get(family, 0.0))
        if weight <= 0:
            continue
        importance = _importance(models[family], sample, features)
        family_methods[family] = str(importance.get("method"))
        for item in importance.get("features", []):
            feature = str(item.get("feature"))
            aggregate[feature] = aggregate.get(feature, 0.0) + weight * float(item.get("importance") or 0.0)
    total = sum(aggregate.values()) or 1.0
    return {
        "method": "exp34_weighted_family_global_importance",
        "scope": "global_training_sample",
        "blend_weights": {name: float(weights.get(name, 0.0)) for name in FAMILIES},
        "family_methods": family_methods,
        "features": [
            {"feature": name, "importance": round(value / total, 6)}
            for name, value in sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
        ],
    }


def _shared_delay_evaluation_rows(test: pd.DataFrame) -> pd.DataFrame:
    """Use the exact verified production Cost cohort for the headline Delay MAE."""
    return _production_cost_evaluation_rows(test)


def train_window_with_promoted_cost_and_delay(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
) -> dict:
    """Train production Cost exactly as before, then replace only Delay with Exp34."""
    result = train_window_with_promoted_cost(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=artifact_root,
    )
    if data is None:
        raise ValueError(
            "Promoted Delay training requires the frozen supervised frame so Exp34 "
            "uses exactly the same project boundary and evidence as production."
        )

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"
    metadata = dict(result.get("metadata") or {})
    contract = target_feature_contract(metadata)
    cost_features = list(contract["cost"])
    base_delay_features = list(contract["delay"])
    risk_features = list(contract["risk"])
    delay_features = list(dict.fromkeys(base_delay_features + PATH_FEATURES))

    cost_hash_before = file_sha256(target / "cost_model.pkl")
    risk_hash_before = file_sha256(target / "risk_model.pkl")
    cost_model = joblib.load(target / "cost_model.pkl")
    risk_model = joblib.load(target / "risk_model.pkl")

    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    delay_weights, delay_oof = _oof_delay_weights(train, delay_features)
    delay_family_models = _fit_delay_family_models(train, delay_features)
    delay_model = ProductionDelayBlendModel(delay_family_models, delay_weights, delay_features)

    full_delay_prediction = np.maximum(0, delay_model.predict(test[delay_features]))
    full_delay_metrics = _regression_metrics(
        test["actual_delay_days"],
        full_delay_prediction,
        test["sample_weight"],
        test["canonical_project_id"],
    )

    shared_eval = _shared_delay_evaluation_rows(test)
    shared_delay_prediction = np.maximum(0, delay_model.predict(shared_eval[delay_features]))
    delay_metrics = _regression_metrics(
        shared_eval["actual_delay_days"],
        shared_delay_prediction,
        shared_eval["sample_weight"],
        shared_eval["canonical_project_id"],
    )

    shared_projects = int(shared_eval["canonical_project_id"].nunique())
    if (
        training_start == VERIFIED_PRODUCTION_START
        and training_end == VERIFIED_PRODUCTION_END
        and test_end == VERIFIED_PRODUCTION_TEST_END
        and shared_projects != VERIFIED_PRODUCTION_EVIDENCE_PROJECTS
    ):
        raise RuntimeError(
            "Refusing to publish the selected 2001-2021 production run because the "
            f"verified evidence cohort changed: expected {VERIFIED_PRODUCTION_EVIDENCE_PROJECTS} "
            f"projects, found {shared_projects}."
        )

    cost_metrics, validation_rows, cost_evaluation_contract = _prediction_rows(
        test,
        cost_model=cost_model,
        cost_features=cost_features,
        delay_model=delay_model,
        delay_features=delay_features,
        risk_model=risk_model,
        risk_features=risk_features,
    )

    prior_cost_mae = ((result.get("lifecycle") or {}).get("metrics") or {}).get("cost", {}).get("MAE")
    if prior_cost_mae is not None and abs(float(prior_cost_mae) - float(cost_metrics["MAE"])) > 1e-12:
        raise AssertionError(
            f"Exp34 Delay promotion changed production Cost MAE: before={prior_cost_mae}, after={cost_metrics['MAE']}"
        )

    joblib.dump(delay_model, target / "delay_model.pkl")
    validation_rows.to_csv(target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")

    if file_sha256(target / "cost_model.pkl") != cost_hash_before:
        raise AssertionError("Exp34 Delay promotion modified cost_model.pkl unexpectedly.")
    if file_sha256(target / "risk_model.pkl") != risk_hash_before:
        raise AssertionError("Exp34 Delay promotion modified risk_model.pkl unexpectedly.")

    importance_path = target / "shap_importance.json"
    importance = json.loads(importance_path.read_text()) if importance_path.exists() else {}
    importance["delay"] = _delay_importance(delay_family_models, delay_weights, train, delay_features)
    importance_path.write_text(json.dumps(importance, indent=2, allow_nan=False))

    lifecycle_stages = _stage_metrics(validation_rows)
    balanced_stage = _balanced_stage_summary(lifecycle_stages)
    delay_evaluation_contract = {
        "cohort": PRODUCTION_DELAY_EVALUATION_COHORT,
        "source_filter": cost_evaluation_contract.get("cohort"),
        "weighting_policy": "project-balanced after shared Exp12 comparable-cohort filter",
        "test_projects": shared_projects,
        "test_snapshots": int(len(shared_eval)),
        "full_holdout_projects": int(test["canonical_project_id"].nunique()),
        "full_holdout_snapshots": int(len(test)),
        "full_holdout_delay_metrics": full_delay_metrics,
        "verified_2001_2021_project_count": VERIFIED_PRODUCTION_EVIDENCE_PROJECTS,
    }

    metadata["delay_features_used"] = delay_features
    metadata["feature_count_by_target"] = {
        "cost": len(cost_features),
        "delay": len(delay_features),
        "risk": len(risk_features),
    }
    metadata["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    metadata["promoted_delay_from_experiment"] = PROMOTED_DELAY_EXPERIMENT_ID
    metadata["delay_policy"] = "exp34_path_dependence_plus_rolling_oof_ensemble"
    metadata["delay_path_features"] = list(PATH_FEATURES)
    metadata["delay_blend_families"] = list(FAMILIES)
    metadata["delay_blend_weights"] = delay_weights
    metadata["delay_rolling_oof"] = delay_oof
    metadata["delay_evaluation_contract"] = delay_evaluation_contract
    metadata["lifecycle_stage_metrics_scope"] = "full_holdout_diagnostic"
    metadata.setdefault("lifecycle_metrics", {})["cost"] = cost_metrics
    metadata.setdefault("lifecycle_metrics", {})["delay"] = delay_metrics
    metadata["delay_full_holdout_metrics"] = full_delay_metrics
    metadata["lifecycle_stage_metrics"] = lifecycle_stages
    metadata["balanced_stage_summary"] = balanced_stage
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["delay"] = "exp34_oof_blend"
    metadata["selected_algorithms"] = selected
    metadata.setdefault("hyperparameters", {})["delay"] = {
        "blend_weights": delay_weights,
        "families": {
            family: delay_family_models[family].named_steps["model"].get_params()
            for family in FAMILIES
        },
    }
    metadata["leakage_policy"] = (
        str(metadata.get("leakage_policy") or "")
        + " Production Delay uses Exp34 cumulative path features constructed only from the "
        "current/prior official snapshots for each project; ensemble weights are selected "
        "only from rolling folds inside the training window; future holdout projects are "
        "never used for feature or blend selection."
    ).strip()

    provenance = dict(metadata.get("provenance") or {})
    provenance["feature_schema_fingerprint"] = feature_schema_fingerprint(
        list(dict.fromkeys(cost_features + delay_features + risk_features))
    )
    provenance["artifact_fingerprints"] = artifact_fingerprints(target, _FINGERPRINTED_ARTIFACTS)
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
    lifecycle["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    lifecycle["delay_blend_weights"] = delay_weights
    lifecycle["delay_evaluation_contract"] = delay_evaluation_contract
    result["lifecycle"] = lifecycle

    promotion = dict(result.get("promotion") or {})
    promotion["delay_retained"] = False
    promotion["delay_experiment_id"] = PROMOTED_DELAY_EXPERIMENT_ID
    promotion["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    promotion["delay_blend_weights"] = delay_weights
    promotion["delay_evaluation_contract"] = delay_evaluation_contract
    result["promotion"] = promotion

    result = _json_safe(result)
    metadata = result["metadata"]
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (target / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
