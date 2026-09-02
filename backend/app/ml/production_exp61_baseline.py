"""Production promotion for verified Experiment 61.

Cost: Exp51 fold-stable shrunk residual calibration on the existing Exp12 raw
Cost model. Delay: Exp58 normalized taxonomy + strictly historical hierarchical
Delay priors added to the current Exp32 AFT route, with the existing Exp34
fallback retained. Risk is unchanged.
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
    _delay_aft_calibration_oof,
    _delay_from_remaining,
    _fit_aft_family_models,
    _remaining_frame,
)
from backend.app.ml.experiments.nextgen_common import (
    _apply_prior,
    _cost_oof,
    _family,
    _prepare,
    _prior_map,
    normalize_taxonomy,
    shrunk_calibration,
)
from backend.app.ml.monthly_training import MODEL_ROOT, _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _prediction_rows, _production_cost_evaluation_rows, target_feature_contract
from backend.app.ml.production_exp35_baseline import (
    AFTResidualDelayModel,
    CALIBRATION_GATE_FEATURE,
    ResidualCalibratedCostModel,
    _select_aft_calibration_projects,
    train_window_with_promoted_cost_and_delay as train_previous_production,
)
from backend.app.ml.provenance import artifact_fingerprints, feature_schema_fingerprint, file_sha256

PROMOTED_EXPERIMENT_ID = "exp_61"
PRODUCTION_COST_BASELINE = "exp12_plus_exp51_shrunk_residual_v2"
PRODUCTION_DELAY_BASELINE = "exp32_aft_plus_exp58_hierarchical_prior_exp33_calibration_exp34_fallback_v3"
SHRINKAGE_STRENGTH = 40.0
_FINGERPRINTED_ARTIFACTS = [
    "cost_model.pkl", "delay_model.pkl", "risk_model.pkl", "feature_quality_report.json",
    "shap_importance.json", "prediction_validation.csv",
]


def _norm_value(value) -> str:
    if pd.isna(value):
        return "<NA>"
    text = str(value).lower()
    import re
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text or "<NA>"


def _serialize_prior_maps(global_value: float, maps: dict) -> dict:
    state = {"global": float(global_value), "levels": []}
    for keys in [("_norm_implementing_agency", "_norm_sector"), ("_norm_sector",), ("_norm_implementing_agency",)]:
        table = maps[keys]
        values = {}
        for _, row in table.iterrows():
            key = tuple(str(row[k]) for k in keys)
            values[key] = (float(row["value"]), float(row["count"]))
        state["levels"].append((tuple(keys), values))
    return state


def _apply_serialized_prior(frame: pd.DataFrame, state: dict) -> tuple[np.ndarray, np.ndarray]:
    agency = frame.get("implementing_agency", pd.Series(pd.NA, index=frame.index)).map(_norm_value)
    sector = frame.get("sector", pd.Series(pd.NA, index=frame.index)).map(_norm_value)
    normalized = frame.copy()
    normalized["_norm_implementing_agency"] = agency
    normalized["_norm_sector"] = sector
    result = np.full(len(normalized), float(state["global"]), dtype=float)
    support = np.zeros(len(normalized), dtype=float)
    unresolved = np.ones(len(normalized), dtype=bool)
    for keys, mapping in state["levels"]:
        for pos, (_, row) in enumerate(normalized.iterrows()):
            if not unresolved[pos]:
                continue
            key = tuple(str(row[k]) for k in keys)
            item = mapping.get(key)
            if item is not None:
                result[pos], support[pos] = float(item[0]), float(item[1])
                unresolved[pos] = False
    return result, support


class Exp61PriorAFTDelayModel:
    """Persisted Exp58-prior AFT route with exact Exp34 fallback."""

    def __init__(self, *, aft_models, weights, base_features, model_features, calibration, fallback_model, prior_state):
        self.aft_models = aft_models
        self.weights = {k: float(v) for k, v in weights.items()}
        self.base_features = list(base_features)
        self.model_features = list(model_features)
        self.features = list(base_features)
        self.calibration = calibration
        self.fallback_model = fallback_model
        self.prior_state = prior_state

    def _enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.copy()
        prior, support = _apply_serialized_prior(work, self.prior_state)
        work["exp58_delay_hier_prior"] = prior
        work["exp58_group_support"] = support
        return work

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = self._enrich(frame)
        prediction = np.maximum(0.0, np.asarray(self.fallback_model.predict(work.reindex(columns=self.base_features)), dtype=float))
        eligible = AFTResidualDelayModel._aft_eligible(work).to_numpy(dtype=bool)
        if not eligible.any():
            return prediction
        positions = np.flatnonzero(eligible)
        subset = work.iloc[positions].copy()
        remaining = _aft_remaining_prediction(self.aft_models, self.weights, subset, self.model_features)
        raw = _delay_from_remaining(subset, remaining)
        prediction[positions] = np.maximum(0.0, raw + _corrections(subset, raw, self.calibration))
        return prediction


def _build_temporal_delay_priors(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = normalize_taxonomy(train.copy())
    test = normalize_taxonomy(test.copy())
    projects = train.groupby("canonical_project_id", as_index=False).agg(
        completion_year=("completion_year", "first"),
        actual_delay_days=("actual_delay_days", "first"),
        _norm_implementing_agency=("_norm_implementing_agency", "first"),
        _norm_sector=("_norm_sector", "first"),
    )
    projects["completion_year"] = pd.to_numeric(projects["completion_year"], errors="coerce")
    train["exp58_delay_hier_prior"] = 0.0
    train["exp58_group_support"] = 0.0
    for year in sorted(projects["completion_year"].dropna().astype(int).unique()):
        prior_projects = projects[projects["completion_year"] < year]
        mask = pd.to_numeric(train["completion_year"], errors="coerce").eq(year)
        if prior_projects.empty or not mask.any():
            continue
        global_value, maps = _prior_map(prior_projects, "actual_delay_days")
        values, support = _apply_prior(train.loc[mask], global_value, maps)
        train.loc[mask, "exp58_delay_hier_prior"] = values
        train.loc[mask, "exp58_group_support"] = support
    global_value, maps = _prior_map(projects, "actual_delay_days")
    values, support = _apply_prior(test, global_value, maps)
    test["exp58_delay_hier_prior"] = values
    test["exp58_group_support"] = support
    return train, test, _serialize_prior_maps(global_value, maps)


def _gain(base: float, candidate: float) -> float:
    return (float(base) - float(candidate)) / float(base) * 100.0 if float(base) else 0.0


def train_window_with_promoted_cost_and_delay(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
) -> dict:
    result = train_previous_production(training_start, training_end, test_end, data=data, identity=identity, artifact_root=artifact_root)
    if data is None:
        raise ValueError("Exp61 production promotion requires the frozen supervised frame")

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"
    metadata = dict(result.get("metadata") or {})
    contract = target_feature_contract(metadata)
    old_cost_features = list(contract["cost"])
    old_delay_support_features = list(contract["delay"])
    risk_features = list(contract["risk"])

    current_cost = joblib.load(target / "cost_model.pkl")
    current_delay = joblib.load(target / "delay_model.pkl")
    risk_hash_before = file_sha256(target / "risk_model.pkl")

    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, training_start, training_end, test_end)

    # Exp51 Cost: replace only the Exp33 calibration with the verified shrunk variant.
    cost_base_features = list(current_cost.features)
    cost_family = _family(current_cost)
    cost_calibration = shrunk_calibration(_cost_oof(train, cost_base_features, cost_family), SHRINKAGE_STRENGTH)
    cost_model = ResidualCalibratedCostModel(current_cost.model, cost_base_features, cost_calibration)

    # Exp58 Delay: strict earlier-year priors for training; full training priors for future rows.
    prior_train, prior_test, prior_state = _build_temporal_delay_priors(train, test)
    delay_base_features = list(current_delay.features)
    delay_model_features = list(dict.fromkeys(delay_base_features + ["exp58_delay_hier_prior", "exp58_group_support"]))
    train_delay = _remaining_frame(prior_train)
    delay_calibration, delay_oof = _delay_aft_calibration_oof(train_delay, delay_model_features, current_delay.weights)
    aft_models = _fit_aft_family_models(train_delay, delay_model_features)
    delay_model = Exp61PriorAFTDelayModel(
        aft_models=aft_models,
        weights=current_delay.weights,
        base_features=delay_base_features,
        model_features=delay_model_features,
        calibration=delay_calibration,
        fallback_model=current_delay.fallback_model,
        prior_state=prior_state,
    )

    shared_eval = _production_cost_evaluation_rows(prior_test).copy()
    calibration_project_ids = _select_aft_calibration_projects(shared_eval)
    shared_eval[CALIBRATION_GATE_FEATURE] = shared_eval["canonical_project_id"].astype("string").isin(calibration_project_ids)
    test = prior_test.copy()
    test[CALIBRATION_GATE_FEATURE] = test["canonical_project_id"].astype("string").isin(calibration_project_ids)

    old_cost_prediction = np.asarray(current_cost.predict(shared_eval), dtype=float)
    old_delay_prediction = np.maximum(0.0, np.asarray(current_delay.predict(shared_eval), dtype=float))
    new_cost_prediction = np.asarray(cost_model.predict(shared_eval), dtype=float)
    new_delay_prediction = np.maximum(0.0, np.asarray(delay_model.predict(shared_eval), dtype=float))

    def metric(actual, pred):
        return _regression_metrics(shared_eval[actual], pred, shared_eval["sample_weight"], shared_eval["canonical_project_id"])

    old_cost_metrics = metric("actual_cost_overrun_percentage", old_cost_prediction)
    old_delay_metrics = metric("actual_delay_days", old_delay_prediction)
    cost_metrics = metric("actual_cost_overrun_percentage", new_cost_prediction)
    delay_metrics = metric("actual_delay_days", new_delay_prediction)

    # The reference decision window still has performance guards, but its cohort
    # size is whatever the evidence rule yields rather than a hard-coded count.
    if (training_start, training_end, test_end) == (2001, 2021, 2025):
        if float(cost_metrics["MAE"]) >= float(old_cost_metrics["MAE"]):
            raise RuntimeError("Exp61 Cost failed to improve the verified production window")
        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):
            raise RuntimeError("Exp61 Delay failed to improve the verified production window")

    delay_support_features = list(dict.fromkeys(old_delay_support_features + ["sector", "implementing_agency"]))
    computed_cost_metrics, validation_rows, cost_evaluation_contract = _prediction_rows(
        test,
        cost_model=cost_model,
        cost_features=old_cost_features,
        delay_model=delay_model,
        delay_features=delay_support_features,
        risk_model=joblib.load(target / "risk_model.pkl"),
        risk_features=risk_features,
    )
    if abs(float(computed_cost_metrics["MAE"]) - float(cost_metrics["MAE"])) > 1e-9:
        raise AssertionError("Exp61 persisted Cost evaluation disagrees with shared cohort")

    joblib.dump(cost_model, target / "cost_model.pkl")
    joblib.dump(delay_model, target / "delay_model.pkl")
    validation_rows.to_csv(target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
    if file_sha256(target / "risk_model.pkl") != risk_hash_before:
        raise AssertionError("Exp61 promotion modified Risk unexpectedly")

    metadata["base_production_cost_baseline"] = metadata.get("production_cost_baseline")
    metadata["base_production_delay_baseline"] = metadata.get("production_delay_baseline")
    metadata["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    metadata["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    metadata["promoted_from_experiment"] = PROMOTED_EXPERIMENT_ID
    metadata["promoted_delay_from_experiment"] = PROMOTED_EXPERIMENT_ID
    metadata["promotion_scope"] = "cost+delay"
    metadata["cost_policy"] = "exp51_fold_stable_shrunk_residual_calibration_strength_40"
    metadata["delay_policy"] = "exp58_strict_temporal_hierarchical_delay_priors_on_exp32_aft_with_exp34_fallback"
    metadata["risk_policy"] = "existing_production_retained"
    metadata["cost_features_used"] = old_cost_features
    metadata["delay_features_used"] = delay_support_features
    metadata["risk_features_used"] = risk_features
    metadata["feature_count_by_target"] = {"cost": len(old_cost_features), "delay": len(delay_support_features), "risk": len(risk_features)}
    metadata["cost_exp51_shrinkage_strength"] = SHRINKAGE_STRENGTH
    metadata["delay_exp58_model_features"] = delay_model_features
    metadata["delay_exp58_prior_state_support"] = {"global": prior_state["global"], "levels": [list(keys) for keys, _ in prior_state["levels"]]}
    metadata["delay_aft_rolling_oof"] = delay_oof
    metadata["cost_evaluation_contract"] = cost_evaluation_contract
    metadata.setdefault("lifecycle_metrics", {})["cost"] = cost_metrics
    metadata.setdefault("lifecycle_metrics", {})["delay"] = delay_metrics
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["cost"] = "exp12_plus_exp51_shrunk_residual"
    selected["delay"] = "exp32_aft_plus_exp58_hierarchical_prior_with_exp34_fallback"
    metadata["selected_algorithms"] = selected
    metadata["leakage_policy"] = (str(metadata.get("leakage_policy") or "") + " Exp61 Cost shrinkage is learned only from rolling training OOF residuals. Exp58 Delay training priors use only projects completed in strictly earlier years; future inference priors are frozen from the allowed training population. Holdout outcomes and errors are never used to fit priors, calibration, routing, or model weights.").strip()

    provenance = dict(metadata.get("provenance") or {})
    provenance["feature_schema_fingerprint"] = feature_schema_fingerprint(list(dict.fromkeys(old_cost_features + delay_support_features + risk_features)))
    provenance["artifact_fingerprints"] = artifact_fingerprints(target, _FINGERPRINTED_ARTIFACTS)
    metadata["provenance"] = provenance

    lifecycle = dict(result.get("lifecycle") or {})
    lifecycle.setdefault("metrics", {})["cost"] = cost_metrics
    lifecycle.setdefault("metrics", {})["delay"] = delay_metrics
    lifecycle["target_features"] = {"cost": old_cost_features, "delay": delay_support_features, "risk": risk_features}
    lifecycle["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    lifecycle["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    result["metadata"] = metadata
    result["lifecycle"] = lifecycle
    result["promotion"] = {
        "experiment_id": PROMOTED_EXPERIMENT_ID,
        "scope": "cost+delay",
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "production_delay_baseline": PRODUCTION_DELAY_BASELINE,
        "previous_cost_mae": old_cost_metrics["MAE"],
        "promoted_cost_mae": cost_metrics["MAE"],
        "cost_improvement_percentage": round(_gain(old_cost_metrics["MAE"], cost_metrics["MAE"]), 6),
        "previous_delay_mae": old_delay_metrics["MAE"],
        "promoted_delay_mae": delay_metrics["MAE"],
        "delay_improvement_percentage": round(_gain(old_delay_metrics["MAE"], delay_metrics["MAE"]), 6),
        "risk_retained": True,
    }
    result = _json_safe(result)
    (target / "metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False))
    (target / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
