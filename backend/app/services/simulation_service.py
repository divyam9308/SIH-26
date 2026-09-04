"""API-facing helpers for real PAIMANA historical model simulations."""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from backend.app.ml.real_time_windows import (
    FEATURES,
    RISK_LEVELS,
    _classifier,
    _fit_classifier,
    _fit_regressor,
    _predict_regressor,
    _regressor,
    evaluate,
    add_leave_one_out_training_priors,
    apply_historical_priors,
    historical_prior_maps,
    labelled,
    model_dir,
    outcome_data,
    versions,
    version_key,
    predict_quantiles,
    scale_quantile_range,
)

# Live judge demo sessions are intentionally process-local. They contain fitted model
# objects and held-out rows, while API responses keep actual outcomes server-side until
# the explicit reveal step.
_CUSTOM_SESSIONS: dict[str, dict] = {}
_MAX_CUSTOM_SESSIONS = 20


def _value(value: Any):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 4)
    return value


def available_versions() -> list[dict]:
    return versions()


def available_data_years() -> list[dict]:
    frame = labelled(outcome_data())
    counts = frame.groupby("completion_year").size().sort_index()
    return [{"year": int(year), "completed_projects": int(count)} for year, count in counts.items()]


def _shap_factors_for_model(model, row: pd.Series, feature_names: list[str] | None = None) -> list[dict]:
    """Return local SHAP contributions without exposing target fields."""
    try:
        names_used = feature_names or FEATURES
        if model.__class__.__name__ == "ShiftedLogCostModel":
            model = model.model
        if hasattr(model, "severity_model"):
            model = model.severity_model
        elif hasattr(model, "models") and model.models:
            model = model.models[0]
        if model.__class__.__module__.startswith("catboost"):
            categorical = [index for index, name in enumerate(names_used) if name in {"sector", "ministry", "implementing_agency", "state", "project_size_category"}]
            one_row = pd.DataFrame([row[names_used]])
            shap_values = np.asarray(model.get_feature_importance(Pool(one_row, cat_features=categorical), type="ShapValues"), dtype=float)
            if shap_values.ndim == 3:
                predicted_class = int(np.asarray(model.predict(one_row), dtype=int).reshape(-1)[0])
                values = shap_values[0, predicted_class, :-1]
            else:
                values = shap_values[0, :-1]
            return [{"feature": feature, "impact": round(float(impact), 4), "direction": "increases" if impact >= 0 else "reduces"} for feature, impact in sorted(zip(names_used, values), key=lambda item: abs(item[1]), reverse=True)[:5]]
        import shap

        transformed = model.named_steps["preprocess"].transform(pd.DataFrame([row[names_used]]))
        values = shap.TreeExplainer(model.named_steps["model"]).shap_values(
            transformed.toarray() if hasattr(transformed, "toarray") else transformed
        )
        names = model.named_steps["preprocess"].get_feature_names_out()
        values = np.asarray(values)[0]
        result = []
        for name, impact in sorted(zip(names, values), key=lambda item: abs(item[1]), reverse=True)[:5]:
            clean_name = str(name).replace("numeric__", "").replace("category__", "")
            for categorical in ("sector", "ministry", "implementing_agency", "state", "project_size_category"):
                if clean_name.startswith(f"{categorical}_"):
                    clean_name = categorical
                    break
            result.append(
                {
                    "feature": clean_name,
                    "impact": round(float(impact), 4),
                    "direction": "increases" if impact >= 0 else "reduces",
                }
            )
        return result
    except Exception:
        return [{"feature": "approved_cost_cr", "impact": 0.0, "direction": "not available"}]


def _shap_factors(key: str, row: pd.Series) -> list[dict]:
    target = model_dir(key)
    metadata = __import__("json").loads((target / "metadata.json").read_text())
    return _shap_factors_for_model(joblib.load(target / "cost_model.pkl"), row, metadata.get("features_used", FEATURES))


def run(key: str) -> dict:
    result = evaluate(key, save=False)
    frame = result["rows"].reset_index(drop=True)
    items = []
    for index, row in frame.iterrows():
        items.append(
            {
                "record_index": int(index),
                "project_id": _value(row.project_id) or "Not published",
                "project_name": _value(row.project_name),
                "sector": _value(row.sector),
                "completion_date": _value(row.completion_date),
                "approved_cost_cr": _value(row.approved_cost_cr),
                "predicted_cost_overrun": _value(row.predicted_cost_overrun),
                "actual_cost_overrun": _value(row.actual_cost_overrun),
                "cost_error": _value(row.cost_error),
                "predicted_delay_days": _value(row.predicted_delay_days),
                "actual_delay_days": _value(row.actual_delay_days),
                "delay_error": _value(row.delay_error),
                "predicted_risk": RISK_LEVELS[int(row.predicted_risk)],
                "actual_risk": RISK_LEVELS[int(row.actual_risk)],
                "snapshot": {
                    "approved_cost_cr": _value(row.approved_cost_cr),
                    "current_cost_cr": None,
                    "physical_progress_percentage": None,
                    "expenditure_cr": None,
                    "sector": _value(row.sector),
                    "note": "The completed-project archive reports approved cost and completion expenditure. Pre-completion current cost, progress, and expenditure were not reported in this source row.",
                },
                "shap_explanation": _shap_factors(key, row),
            }
        )
    return {
        "version": key,
        "metrics": result["metrics"],
        "items": items,
        "reveal_policy": "Actual outcome values are present for the historical evaluation response but should remain hidden until the user selects Reveal Actual Outcome.",
    }


def _fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        "project_name",
        "approved_cost_cr",
        "planned_commissioning_date",
        "reported_completion_expenditure_cr",
        "completion_date",
        "source_url",
    ]
    canonical = frame[columns].sort_values(columns[:2], kind="stable").to_csv(index=False, date_format="%Y-%m-%d")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def train_custom(start_year: int, end_year: int) -> dict:
    """Train a fresh model and hold every later completed project out of fitting."""
    all_data = labelled(outcome_data()).sort_values(["completion_year", "project_name"], kind="stable").reset_index(drop=True)
    if start_year > end_year:
        raise ValueError("Training start year must be less than or equal to training end year.")

    min_year = int(all_data.completion_year.min())
    max_year = int(all_data.completion_year.max())
    if end_year >= max_year:
        raise ValueError(f"Training must end before {max_year} so at least one later held-out year remains for testing.")
    if end_year < min_year or start_year > max_year:
        raise ValueError(f"Training range must overlap available official completed-project data ({min_year}-{max_year}).")

    train_data = all_data[all_data.completion_year.between(start_year, end_year)].copy()
    if len(train_data) < 12:
        raise ValueError(f"Selected range has only {len(train_data)} eligible completed projects; choose a wider range with at least 12.")

    held_out = all_data[all_data.completion_year > end_year].copy().reset_index(drop=True)
    if held_out.empty:
        raise ValueError("No later completed projects are available for a leakage-free historical test.")

    target = model_dir(version_key(start_year, end_year))
    metadata_path = target / "metadata.json"
    compatible_registry = False
    if metadata_path.exists():
        registry_metadata = __import__("json").loads(metadata_path.read_text())
        compatible_registry = registry_metadata.get("features_used") == FEATURES
    if compatible_registry and all((target / name).exists() for name in ("cost_model.pkl", "delay_model.pkl", "risk_model.pkl")):
        cost = joblib.load(target / "cost_model.pkl")
        delay = joblib.load(target / "delay_model.pkl")
        risk = joblib.load(target / "risk_model.pkl")
        uncertainty = joblib.load(target / "uncertainty_models.pkl") if (target / "uncertainty_models.pkl").exists() else None
        calibration_path = target / "confidence_calibration.json"
        calibration = __import__("json").loads(calibration_path.read_text()) if calibration_path.exists() else None
        priors_path = target / "historical_priors.json"
        if priors_path.exists():
            held_out = apply_historical_priors(held_out, __import__("json").loads(priors_path.read_text()))
    else:
        priors = historical_prior_maps(train_data)
        held_out = apply_historical_priors(held_out, priors)
        train_data = add_leave_one_out_training_priors(train_data)
        X = train_data[FEATURES]
        cost = _regressor()
        delay = _regressor(seed=26104)
        risk = _classifier(train_data.actual_risk)
        _fit_regressor(cost, X, train_data.actual_cost_overrun_percentage)
        _fit_regressor(delay, X, train_data.actual_delay_days, delay_target=True)
        _fit_classifier(risk, X, train_data.actual_risk)
        uncertainty = None
        calibration = None

    held_out["record_index"] = np.arange(len(held_out), dtype=int)
    session_id = uuid.uuid4().hex[:16]
    _CUSTOM_SESSIONS[session_id] = {
        "training_start": int(start_year),
        "training_end": int(end_year),
        "training_samples": int(len(train_data)),
        "training_fingerprint": _fingerprint(train_data),
        "cost_model": cost,
        "delay_model": delay,
        "risk_model": risk,
        "uncertainty_models": uncertainty,
        "confidence_calibration": calibration,
        "held_out": held_out,
        "predictions": {},
    }
    while len(_CUSTOM_SESSIONS) > _MAX_CUSTOM_SESSIONS:
        oldest = next(iter(_CUSTOM_SESSIONS))
        if oldest == session_id and len(_CUSTOM_SESSIONS) == 1:
            break
        _CUSTOM_SESSIONS.pop(oldest, None)

    year_counts = held_out.groupby("completion_year").size().sort_index()
    return {
        "session_id": session_id,
        "training_start": int(start_year),
        "training_end": int(end_year),
        "training_samples": int(len(train_data)),
        "training_fingerprint_sha256": _CUSTOM_SESSIONS[session_id]["training_fingerprint"],
        "features_used": FEATURES,
        "data_source": "Official PAIMANA completed-project archive reports",
        "eligible_test_years": [{"year": int(year), "projects": int(count)} for year, count in year_counts.items()],
        "leakage_guard": f"Only projects completed in {start_year}-{end_year} were fitted. Test projects must be completed after {end_year}.",
        "actual_outcomes_sent_to_browser": False,
    }


def _session(session_id: str) -> dict:
    if session_id not in _CUSTOM_SESSIONS:
        raise KeyError(session_id)
    return _CUSTOM_SESSIONS[session_id]


def custom_projects(session_id: str, year: int) -> dict:
    session = _session(session_id)
    if year <= session["training_end"]:
        raise ValueError(f"Test year must be after the training cutoff ({session['training_end']}).")
    frame = session["held_out"]
    rows = frame[frame.completion_year == year]
    if rows.empty:
        raise ValueError(f"No held-out official completed projects are available for {year}.")
    items = []
    for _, row in rows.iterrows():
        items.append(
            {
                "record_index": int(row.record_index),
                "project_id": _value(row.project_id) or "Not published",
                "project_name": _value(row.project_name),
                "sector": _value(row.sector),
                "implementing_agency": _value(row.implementing_agency),
                "approved_cost_cr": _value(row.approved_cost_cr),
                "planned_commissioning_year": _value(row.planned_commissioning_year),
                "held_out_completion_year": int(row.completion_year),
            }
        )
    return {
        "session_id": session_id,
        "year": int(year),
        "items": items,
        "actual_outcomes_sent_to_browser": False,
        "note": "The held-out completion year is used only to select an unseen historical project; completion date and final expenditure are not model inputs.",
    }


def _session_row(session: dict, record_index: int) -> pd.Series:
    frame = session["held_out"]
    rows = frame[frame.record_index == record_index]
    if rows.empty:
        raise ValueError("Selected held-out project does not exist in this training session.")
    return rows.iloc[0]


def predict_custom(session_id: str, record_index: int) -> dict:
    session = _session(session_id)
    row = _session_row(session, record_index)
    X = pd.DataFrame([row[FEATURES]])
    predicted_cost = float(_predict_regressor(session["cost_model"], X)[0])
    predicted_delay = float(_predict_regressor(session["delay_model"], X, delay_target=True)[0])
    predicted_risk = int(np.asarray(session["risk_model"].predict(X), dtype=int).reshape(-1)[0])
    probabilities = np.asarray(session["risk_model"].predict_proba(X), dtype=float)[0] if hasattr(session["risk_model"], "predict_proba") else np.array([1.0])
    prediction = {
        "predicted_cost_overrun": round(predicted_cost, 4),
        "predicted_delay_days": round(predicted_delay, 4),
        "predicted_risk": RISK_LEVELS[predicted_risk],
        "risk_probability_percentage": round(float(probabilities.max()) * 100, 1),
    }
    if session.get("uncertainty_models"):
        cost_range = predict_quantiles(session["uncertainty_models"]["cost"], X, delay_target=False)
        delay_range = predict_quantiles(session["uncertainty_models"]["delay"], X, delay_target=True)
        calibration = session.get("confidence_calibration") or {}
        cost_range = scale_quantile_range(cost_range, float(calibration.get("cost", {}).get("scale", 1.0)))
        delay_range = scale_quantile_range(delay_range, float(calibration.get("delay", {}).get("scale", 1.0)))
        prediction["expected_range"] = {
            "cost_overrun_percentage": {label: round(float(values[0]), 4) for label, values in cost_range.items()},
            "delay_days": {label: round(float(values[0]), 4) for label, values in delay_range.items()},
        }
        prediction["model_confidence_percentage"] = round(float(calibration.get("confidence_percentage", 0.0)), 1)
        prediction["confidence_calibration_status"] = calibration.get("status", "unavailable")
    session["predictions"][int(record_index)] = prediction
    return {
        "session_id": session_id,
        "record_index": int(record_index),
        "project": {
            "project_id": _value(row.project_id) or "Not published",
            "project_name": _value(row.project_name),
            "sector": _value(row.sector),
            "implementing_agency": _value(row.implementing_agency),
        },
        "model_inputs": {feature: _value(row.get(feature)) for feature in FEATURES},
        **prediction,
        "shap_explanation": _shap_factors_for_model(session["cost_model"], row),
        "audit": {
            "training_start": session["training_start"],
            "training_end": session["training_end"],
            "training_samples": session["training_samples"],
            "training_fingerprint_sha256": session["training_fingerprint"],
            "project_excluded_from_training": int(row.completion_year) > session["training_end"],
            "actual_outcomes_sent_to_browser": False,
        },
    }


def reveal_custom(session_id: str, record_index: int) -> dict:
    session = _session(session_id)
    if int(record_index) not in session["predictions"]:
        raise ValueError("Generate the AI prediction before revealing the official outcome.")
    row = _session_row(session, record_index)
    predicted = session["predictions"][int(record_index)]
    actual_cost = float(row.actual_cost_overrun_percentage)
    actual_delay = float(row.actual_delay_days)
    actual_risk = RISK_LEVELS[int(row.actual_risk)]
    return {
        "session_id": session_id,
        "record_index": int(record_index),
        "actual_cost_overrun": round(actual_cost, 4),
        "actual_delay_days": round(actual_delay, 4),
        "actual_risk": actual_risk,
        "cost_error_absolute_pp": round(abs(predicted["predicted_cost_overrun"] - actual_cost), 4),
        "delay_error_absolute_days": round(abs(predicted["predicted_delay_days"] - actual_delay), 4),
        "completion_date": _value(row.completion_date),
        "reported_completion_expenditure_cr": _value(row.reported_completion_expenditure_cr),
        "source_url": _value(row.source_url),
        "source_label": "Official PAIMANA completed-project archive report",
        "reveal_policy": "Actual fields were returned only after a prediction had been generated for this held-out project.",
    }
