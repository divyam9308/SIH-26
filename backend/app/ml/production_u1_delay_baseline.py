"""Production promotion of U1/Exp62 for Delay only.

The existing Exp61 production stack is trained first. Cost and Risk are retained
byte-for-byte. Delay is wrapped with the exact U1 nonlinear rolling-OOF residual
booster used by Experiment 62: a heavily regularized LightGBM predicts a bounded
correction to the Exp61 Delay forecast from training-only OOF residuals.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _aft_remaining_prediction,
    _corrections,
    _delay_from_remaining,
    _fit_aft_family_models,
    _remaining_frame,
)
from backend.app.ml.experiments.nextgen_common import _compare, _prepare, normalize_taxonomy
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.monthly_training import MODEL_ROOT, _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _prediction_rows, target_feature_contract
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE, _select_aft_calibration_projects
from backend.app.ml.production_exp61_baseline import (
    _build_temporal_delay_priors,
    train_window_with_promoted_cost_and_delay as train_exp61_production,
)
from backend.app.ml.provenance import artifact_fingerprints, feature_schema_fingerprint, file_sha256

PROMOTED_DELAY_EXPERIMENT_ID = "exp_62_u1_delay_only"
PRODUCTION_DELAY_BASELINE = "exp61_plus_u1_nonlinear_oof_residual_booster_v1"
U1_SEED = 6202
U1_CANDIDATES = [
    "production_prediction",
    "cost_escalation_percentage",
    "schedule_slippage_days",
    "duration_ratio",
    "expenditure_ratio",
    "progress_deviation",
    "approved_cost_cr",
    "exp58_delay_hier_prior",
    "exp58_group_support",
]
_RAW_U1_INPUTS = [
    "cost_escalation_percentage",
    "schedule_slippage_days",
    "duration_ratio",
    "expenditure_ratio",
    "progress_deviation",
    "approved_cost_cr",
    "sector",
    "implementing_agency",
]
_FINGERPRINTED_ARTIFACTS = [
    "cost_model.pkl",
    "delay_model.pkl",
    "risk_model.pkl",
    "feature_quality_report.json",
    "shap_importance.json",
    "prediction_validation.csv",
]


def _weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[mask], weights[mask]
    if not len(values):
        return 0.0
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total = float(weights.sum())
    if total <= 0:
        return float(np.quantile(values, q))
    return float(values[np.searchsorted(np.cumsum(weights), q * total, side="left")])


def _design(train: pd.DataFrame, score: pd.DataFrame):
    cols = [c for c in U1_CANDIDATES if c in train.columns and c in score.columns]
    if "production_prediction" not in cols:
        raise AssertionError("U1 Delay production promotion requires production_prediction")
    medians: dict[str, float] = {}
    train_cols = {}
    score_cols = {}
    for col in cols:
        a = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(score[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(a.median()) if a.notna().any() else 0.0
        medians[col] = median
        train_cols[col] = a.fillna(median)
        score_cols[col] = b.fillna(median)
    return cols, medians, pd.DataFrame(train_cols), pd.DataFrame(score_cols)


def _fit_u1_booster(oof: pd.DataFrame, score: pd.DataFrame):
    cols, medians, x_train, x_score = _design(oof, score)
    residual = pd.to_numeric(oof["residual"], errors="coerce").fillna(0.0).to_numpy(float)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    booster = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=12,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=5.0,
        reg_lambda=25.0,
        random_state=U1_SEED,
        verbosity=-1,
    )
    booster.fit(x_train, residual, sample_weight=weight)
    cap = max(float(_weighted_quantile(np.abs(residual), weight, 0.90)), 1e-9)
    correction = np.clip(np.asarray(booster.predict(x_score), dtype=float), -cap, cap)
    return booster, medians, cols, cap, correction


def _delay_oof_frame(train: pd.DataFrame, base_delay_model) -> pd.DataFrame:
    features = list(base_delay_model.model_features)
    train_delay = _remaining_frame(train)
    chunks = []
    for fit, val, year in _rolling_folds(train_delay):
        models = _fit_aft_family_models(fit, features)
        remaining = _aft_remaining_prediction(models, base_delay_model.weights, val, features)
        raw = _delay_from_remaining(val, remaining)
        pred = np.maximum(0.0, raw + _corrections(val, raw, base_delay_model.calibration))
        part = val.copy()
        part["production_prediction"] = pred
        part["residual"] = pd.to_numeric(part["actual_delay_days"], errors="coerce") - pred
        part["oof_year"] = int(year)
        chunks.append(part)
    if len(chunks) < 2:
        raise ValueError("U1 Delay production promotion requires at least two rolling OOF folds")
    return pd.concat(chunks, ignore_index=True)


def _u1_prior_enrich(frame: pd.DataFrame, prior_state: dict) -> pd.DataFrame:
    """Rebuild U1's Exp58 booster inputs exactly as its experiment score frame did.

    U1 was trained/scored on ``normalize_taxonomy(_prepare(...))`` rows carrying
    Exp58 future priors. The Exp61 base model has a separate internal enrichment
    path for its own prediction. Reusing that path for U1's *booster features*
    can change normalized taxonomy keys and therefore the residual correction.
    This helper keeps the Exp61 base prediction untouched while reconstructing
    only the two U1 prior covariates from U1's normalized taxonomy contract.
    """
    work = normalize_taxonomy(frame.copy())
    result = np.full(len(work), float(prior_state["global"]), dtype=float)
    support = np.zeros(len(work), dtype=float)
    unresolved = np.ones(len(work), dtype=bool)
    for keys, mapping in prior_state.get("levels", []):
        keys = tuple(keys)
        for pos, (_, row) in enumerate(work.iterrows()):
            if not unresolved[pos]:
                continue
            key = tuple(str(row[k]) for k in keys)
            item = mapping.get(key)
            if item is not None:
                result[pos], support[pos] = float(item[0]), float(item[1])
                unresolved[pos] = False
    work["exp58_delay_hier_prior"] = result
    work["exp58_group_support"] = support
    return work


class U1DelayResidualProductionModel:
    """Exp61 Delay anchor plus the frozen U1 bounded residual correction."""

    def __init__(
        self,
        *,
        base_model,
        booster,
        booster_features,
        medians,
        correction_cap,
        input_features,
        booster_prior_state,
    ):
        self.base_model = base_model
        self.booster = booster
        self.booster_features = list(booster_features)
        self.medians = {str(k): float(v) for k, v in medians.items()}
        self.correction_cap = float(correction_cap)
        self.features = list(input_features)
        self.booster_prior_state = booster_prior_state
        self.model_features = list(getattr(base_model, "model_features", []))
        self.weights = dict(getattr(base_model, "weights", {}))
        self.calibration = getattr(base_model, "calibration", None)
        self.fallback_model = getattr(base_model, "fallback_model", None)

    def _booster_frame(self, frame: pd.DataFrame, production_prediction: np.ndarray) -> pd.DataFrame:
        if self.booster_prior_state:
            work = _u1_prior_enrich(frame, self.booster_prior_state)
        elif hasattr(self.base_model, "_enrich"):
            work = self.base_model._enrich(frame.copy())
        else:
            work = frame.copy()
        work["production_prediction"] = production_prediction
        columns = {}
        for col in self.booster_features:
            values = pd.to_numeric(work.get(col, pd.Series(np.nan, index=work.index)), errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            columns[col] = values.fillna(self.medians.get(col, 0.0))
        return pd.DataFrame(columns, index=work.index)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        production_prediction = np.maximum(0.0, np.asarray(self.base_model.predict(frame), dtype=float))
        x = self._booster_frame(frame, production_prediction)
        correction = np.asarray(self.booster.predict(x), dtype=float)
        correction = np.clip(correction, -self.correction_cap, self.correction_cap)
        return np.maximum(0.0, production_prediction + correction)


def _metric(frame: pd.DataFrame, actual: str, prediction: np.ndarray) -> dict:
    return _regression_metrics(
        frame[actual], prediction, frame["sample_weight"], frame["canonical_project_id"]
    )


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
    result = train_exp61_production(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=artifact_root,
    )
    if data is None:
        raise ValueError("U1 Delay production promotion requires the frozen supervised frame")

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"
    metadata = dict(result.get("metadata") or {})
    contract = target_feature_contract(metadata)
    cost_features = list(contract["cost"])
    risk_features = list(contract["risk"])
    delay_input_features = list(dict.fromkeys(list(contract["delay"]) + _RAW_U1_INPUTS))

    cost_hash_before = file_sha256(target / "cost_model.pkl")
    risk_hash_before = file_sha256(target / "risk_model.pkl")
    cost_model = joblib.load(target / "cost_model.pkl")
    base_delay_model = joblib.load(target / "delay_model.pkl")
    risk_model = joblib.load(target / "risk_model.pkl")

    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, training_start, training_end, test_end)
    prior_train, prior_test, prior_state = _build_temporal_delay_priors(train, test)
    shared_eval = _compare(prior_test)

    production_delay = np.maximum(0.0, np.asarray(base_delay_model.predict(shared_eval), dtype=float))
    oof = _delay_oof_frame(prior_train, base_delay_model)
    score = shared_eval.copy()
    score["production_prediction"] = production_delay
    booster, medians, booster_features, cap, correction = _fit_u1_booster(oof, score)
    promoted_delay = np.maximum(0.0, production_delay + correction)
    delay_model = U1DelayResidualProductionModel(
        base_model=base_delay_model,
        booster=booster,
        booster_features=booster_features,
        medians=medians,
        correction_cap=cap,
        input_features=delay_input_features,
        booster_prior_state=prior_state,
    )

    # Guard the exact production contract immediately: a serialized/live wrapper
    # must reproduce the U1 score-frame prediction before any artifact is written.
    contract_delay = delay_model.predict(shared_eval.reindex(columns=delay_input_features))
    if not np.allclose(contract_delay, promoted_delay, rtol=0.0, atol=1e-9):
        max_abs = float(np.max(np.abs(contract_delay - promoted_delay)))
        raise AssertionError(f"U1 production wrapper diverged from U1 experiment score path; max abs={max_abs}")

    production_cost = np.asarray(cost_model.predict(shared_eval), dtype=float)
    cost_metrics = _metric(shared_eval, "actual_cost_overrun_percentage", production_cost)
    old_delay_metrics = _metric(shared_eval, "actual_delay_days", production_delay)
    delay_metrics = _metric(shared_eval, "actual_delay_days", promoted_delay)

    if (training_start, training_end, test_end) == (2001, 2021, 2025):
        if float(delay_metrics["MAE"]) >= float(old_delay_metrics["MAE"]):
            raise RuntimeError("U1 Delay failed to improve the verified 2001-2021 production window")

    # Full validation rows need the same evidence-only AFT gate as the shared cohort.
    calibration_project_ids = _select_aft_calibration_projects(shared_eval)
    prior_test = prior_test.copy()
    prior_test[CALIBRATION_GATE_FEATURE] = prior_test["canonical_project_id"].astype("string").isin(calibration_project_ids)

    joblib.dump(delay_model, target / "delay_model.pkl")
    if file_sha256(target / "cost_model.pkl") != cost_hash_before:
        raise AssertionError("U1 Delay promotion modified Cost artifact")
    if file_sha256(target / "risk_model.pkl") != risk_hash_before:
        raise AssertionError("U1 Delay promotion modified Risk artifact")

    computed_cost_metrics, validation_rows, cost_evaluation_contract = _prediction_rows(
        prior_test,
        cost_model=cost_model,
        cost_features=cost_features,
        delay_model=delay_model,
        delay_features=delay_input_features,
        risk_model=risk_model,
        risk_features=risk_features,
    )
    if abs(float(computed_cost_metrics["MAE"]) - float(cost_metrics["MAE"])) > 1e-9:
        raise AssertionError("U1 Delay promotion changed the production Cost evaluation")
    validation_rows.to_csv(target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")

    metadata["base_production_delay_baseline"] = metadata.get("production_delay_baseline")
    metadata["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    metadata["promoted_delay_from_experiment"] = PROMOTED_DELAY_EXPERIMENT_ID
    metadata["promotion_scope"] = "delay"
    metadata["cost_policy"] = metadata.get("cost_policy", "Exp61 Cost retained unchanged")
    metadata["delay_policy"] = "Exp61 Delay plus U1 nonlinear rolling-OOF bounded residual booster"
    metadata["risk_policy"] = "existing_production_retained"
    metadata["cost_features_used"] = cost_features
    metadata["delay_features_used"] = delay_input_features
    metadata["risk_features_used"] = risk_features
    metadata["feature_count_by_target"] = {
        "cost": len(cost_features), "delay": len(delay_input_features), "risk": len(risk_features)
    }
    metadata["u1_delay_booster"] = {
        "seed": U1_SEED,
        "features": booster_features,
        "training_medians": medians,
        "correction_cap_abs_residual_q90": cap,
        "oof_rows": int(len(oof)),
        "booster_prior_source": "normalized_taxonomy_frozen_training_prior_state",
        "base_prediction_replaced": False,
        "holdout_used_for_fit_or_selection": False,
    }
    metadata.setdefault("lifecycle_metrics", {})["cost"] = cost_metrics
    metadata.setdefault("lifecycle_metrics", {})["delay"] = delay_metrics
    metadata["cost_evaluation_contract"] = cost_evaluation_contract
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["delay"] = "exp61_delay_plus_u1_bounded_residual_booster"
    metadata["selected_algorithms"] = selected
    metadata["leakage_policy"] = (
        str(metadata.get("leakage_policy") or "")
        + " U1 Delay residual booster is fitted only to rolling OOF residuals from training years; "
          "its correction cap, imputations, and future hierarchical prior inputs are frozen from training-only state. "
          "Future holdout outcomes are never used."
    ).strip()

    lifecycle = dict(result.get("lifecycle") or {})
    lifecycle.setdefault("metrics", {})["cost"] = cost_metrics
    lifecycle.setdefault("metrics", {})["delay"] = delay_metrics
    lifecycle["target_features"] = {
        "cost": cost_features, "delay": delay_input_features, "risk": risk_features
    }
    lifecycle["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    result["lifecycle"] = lifecycle
    result["promotion"] = {
        "experiment_id": PROMOTED_DELAY_EXPERIMENT_ID,
        "scope": "delay",
        "production_delay_baseline": PRODUCTION_DELAY_BASELINE,
        "previous_delay_mae": old_delay_metrics["MAE"],
        "promoted_delay_mae": delay_metrics["MAE"],
        "delay_improvement_percentage": round(_gain(old_delay_metrics["MAE"], delay_metrics["MAE"]), 6),
        "cost_mae": cost_metrics["MAE"],
        "cost_retained": True,
        "risk_retained": True,
    }

    provenance = dict(metadata.get("provenance") or {})
    provenance["feature_schema_fingerprint"] = feature_schema_fingerprint(
        list(dict.fromkeys(cost_features + delay_input_features + risk_features))
    )
    provenance["artifact_fingerprints"] = artifact_fingerprints(target, _FINGERPRINTED_ARTIFACTS)
    metadata["provenance"] = provenance
    result["metadata"] = metadata

    result = _json_safe(result)
    (target / "metadata.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False))
    (target / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
