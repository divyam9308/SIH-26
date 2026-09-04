"""Current-project forecasts using the active real PAIMANA time-window model."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.real_time_windows import FEATURES, RISK_LEVELS, _predict_regressor, active_version, apply_historical_priors, apply_sector_correction, features, model_dir, predict_quantiles, scale_quantile_range
from backend.app.services.data_service import get_project
from backend.app.services.simulation_service import _shap_factors_for_model


def active_model_signature(version: str) -> str:
    """Key model caches by immutable version plus on-disk artifact identity."""
    target = model_dir(version)
    names = ("metadata.json", "cost_model.pkl", "delay_model.pkl", "risk_model.pkl", "uncertainty_models.pkl", "confidence_calibration.json")
    parts = []
    for name in names:
        path = target / name
        if path.exists():
            stat = path.stat()
            parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


@lru_cache(maxsize=4)
def _active_model_bundle(version: str, model_signature: str) -> dict:
    """Load one immutable versioned model bundle for reuse across forecasts."""
    del model_signature
    target = model_dir(version)
    calibration_path = target / "confidence_calibration.json"
    corrections_path = target / "sector_corrections.json"
    uncertainty_path = target / "uncertainty_models.pkl"
    return {
        "target": target,
        "metadata": json.loads((target / "metadata.json").read_text()),
        "priors": json.loads((target / "historical_priors.json").read_text()) if (target / "historical_priors.json").exists() else None,
        "cost_model": joblib.load(target / "cost_model.pkl"),
        "delay_model": joblib.load(target / "delay_model.pkl"),
        "risk_model": joblib.load(target / "risk_model.pkl"),
        "corrections": json.loads(corrections_path.read_text()) if corrections_path.exists() else None,
        "uncertainty": joblib.load(uncertainty_path) if uncertainty_path.exists() else None,
        "calibration": json.loads(calibration_path.read_text()) if calibration_path.exists() else {},
    }


def _model_inputs(project: pd.Series) -> pd.DataFrame:
    approved_cost = pd.to_numeric(project.get("original_cost_cr"), errors="coerce")
    planned_date = pd.to_datetime(project.get("original_end_date"), errors="coerce")
    if pd.isna(approved_cost) or approved_cost <= 0:
        raise ValueError("Forecast unavailable: approved project cost is missing or not positive.")
    if pd.isna(planned_date):
        raise ValueError("Forecast unavailable: planned completion date is missing.")
    # The current PAIMANA monitoring extract does not publish implementing
    # agency for every row. Use its explicit categorical missing value, not a
    # fabricated numerical fallback.
    raw = pd.DataFrame([{
        "approved_cost_cr": float(approved_cost),
        "sector": str(project.get("sector") or "Not reported"),
        "ministry": str(project.get("ministry") or "Not reported"),
        "implementing_agency": str(project.get("implementing_agency") or "Not reported"),
        "state": str(project.get("state") or "Not reported"),
        "planned_commissioning_date": planned_date,
        "revised_cost_cr": project.get("revised_cost_cr"),
        "current_expenditure_cr": project.get("expenditure_cr"),
        "physical_progress": project.get("physical_progress_pct"),
        "snapshot_date": project.get("snapshot_date"),
        "revised_completion_date": project.get("revised_end_date"),
    }])
    return features(raw)


def _completion_probabilities(target, X: pd.DataFrame, planned: pd.Timestamp) -> list[dict]:
    path = target / "survival_model.pkl"
    if not path.exists():
        return []
    try:
        bundle = joblib.load(path)
        covariates = X[bundle["features"]].copy().fillna(bundle["medians"]).fillna(0)
        values = []
        for year in range(max(pd.Timestamp.today().year, planned.year), max(pd.Timestamp.today().year, planned.year) + 3):
            horizon = max(1, (pd.Timestamp(year=year, month=12, day=31) - planned).days + 1)
            survival = bundle["model"].predict_survival_function(covariates, times=[horizon])
            probability = 1.0 - float(survival.iloc[0, 0])
            values.append({"year": year, "probability_percentage": round(np.clip(probability, 0, 1) * 100, 1)})
        return values
    except Exception:
        return []


def project_forecast(code: str, *, include_explanations: bool = True) -> dict:
    project = get_project(code)
    version = active_version()
    if not version:
        raise ValueError("No real PAIMANA time-window model is available. Retrain a model first.")
    bundle = _active_model_bundle(version, active_model_signature(version))
    target = bundle["target"]
    metadata = bundle["metadata"]
    X = _model_inputs(project)
    if bundle["priors"]:
        X = apply_historical_priors(X, bundle["priors"])
    X = X[metadata.get("features_used", FEATURES)]
    cost_model = bundle["cost_model"]
    delay_model = bundle["delay_model"]
    risk_model = bundle["risk_model"]
    corrections = bundle["corrections"]
    cost = float(apply_sector_correction(_predict_regressor(cost_model, X), X, corrections, "cost")[0])
    # A live forecast reports additional delay days.  The backtest retains
    # signed early/late completion outcomes, while a negative live estimate is
    # represented as no predicted delay rather than a negative delay duration.
    delay = max(0.0, float(apply_sector_correction(_predict_regressor(delay_model, X, delay_target=True), X, corrections, "delay")[0]))
    uncertainty = bundle["uncertainty"]
    calibration = bundle["calibration"]
    expected_range = None
    if uncertainty:
        cost_q = predict_quantiles(uncertainty["cost"], X, delay_target=False)
        delay_q = predict_quantiles(uncertainty["delay"], X, delay_target=True)
        cost_q = scale_quantile_range(cost_q, float(calibration.get("cost", {}).get("scale", 1.0)))
        delay_q = scale_quantile_range(delay_q, float(calibration.get("delay", {}).get("scale", 1.0)))
        expected_range = {
            "cost_overrun_percentage": {label: round(float(values[0]), 2) for label, values in cost_q.items()},
            "delay_days": {label: round(float(values[0]), 1) for label, values in delay_q.items()},
        }
    risk_prediction = int(np.asarray(risk_model.predict(X), dtype=int).reshape(-1)[0])
    probabilities = np.asarray(risk_model.predict_proba(X), dtype=float)[0] if hasattr(risk_model, "predict_proba") else np.array([1.0])
    probability = float(probabilities.max())
    planned = pd.to_datetime(project.original_end_date, errors="coerce")
    progress = pd.to_numeric(project.get("physical_progress_pct"), errors="coerce")
    revised = pd.to_numeric(project.get("revised_cost_cr"), errors="coerce")
    expenditure = pd.to_numeric(project.get("expenditure_cr"), errors="coerce")
    factors = _shap_factors_for_model(cost_model, X.iloc[0]) if include_explanations else []
    delay_factors = _shap_factors_for_model(delay_model, X.iloc[0]) if include_explanations else []
    risk_factors = _shap_factors_for_model(risk_model, X.iloc[0]) if include_explanations else []
    current_status = {
        "snapshot_month": pd.to_datetime(project.snapshot_date).strftime("%Y-%m-%d"),
        "physical_progress_percentage": None if pd.isna(progress) else round(float(progress), 1),
        "current_estimated_cost": None if pd.isna(revised) else round(float(revised), 2),
        "expenditure_cr": None if pd.isna(expenditure) else round(float(expenditure), 2),
        "planned_completion_date": planned.strftime("%Y-%m-%d"),
        "progress_delay_percentage_points": None,
    }
    approved = float(project.original_cost_cr)
    predicted_cost_overrun_amount = approved * cost / 100.0
    predicted_final_cost = approved + predicted_cost_overrun_amount
    predicted_completion = planned + pd.to_timedelta(delay, unit="D")
    return {
        "project_id": str(project.project_code), "project_name": project.project_name,
        "model_version": version,
        "dataset_snapshot_date": pd.to_datetime(project.snapshot_date).strftime("%Y-%m-%d"),
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "current_status": current_status, "predicted_cost_overrun_percentage": round(cost, 2),
        "predicted_cost_overrun_amount_cr": round(predicted_cost_overrun_amount, 2),
        "predicted_final_cost_cr": round(predicted_final_cost, 2),
        "predicted_delay_days": round(delay, 1), "predicted_cost_overrun": round(cost, 2),
        "predicted_completion_date": predicted_completion.strftime("%Y-%m-%d"),
        "current_progress": current_status["physical_progress_percentage"],
        "predicted_delay_months": round(delay / 30.4375, 1),
        "risk_score": round(probability * 100, 1), "risk_probability_percentage": round(probability * 100, 1), "risk_level": RISK_LEVELS[risk_prediction],
        "model_confidence_percentage": round(float(calibration.get("confidence_percentage", 0.0)), 1) if uncertainty else None,
        "confidence_calibration_status": calibration.get("status", "unavailable") if uncertainty else "unavailable",
        "explanation": factors, "shap_explanation": factors, "cost_factors": factors, "delay_factors": delay_factors, "risk_factors": risk_factors,
        "best_models": {"cost": metadata.get("algorithms", {}).get("cost", "registered model"), "delay": metadata.get("algorithms", {}).get("delay", "registered model")},
        "expected_range": expected_range,
        "completion_probabilities": _completion_probabilities(target, X, planned),
        "features_used": metadata.get("features_used", FEATURES),
        "model_scope": f"Real PAIMANA time-window model {version}; final expenditure and completion date are not inference inputs.",
    }


def project_prediction(code: str, override: dict | None = None, include_explanations: bool = True) -> dict:
    return project_forecast(code, include_explanations=include_explanations)


def clear_prediction_caches() -> None:
    """Explicit invalidation hook for model activation/retraining operations."""
    _active_model_bundle.cache_clear()
