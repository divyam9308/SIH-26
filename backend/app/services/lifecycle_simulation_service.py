"""Judge-facing historical simulation backed by freshly retrained lifecycle models."""
from __future__ import annotations

from functools import lru_cache
import json
import math
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import TRAJECTORIES, build_training_dataset
from backend.app.ml.monthly_training import MODEL_ROOT
from backend.app.services.lifecycle_retraining_service import retrain_lifecycle
from backend.app.services.simulation_service import _shap_factors_for_model

_CUSTOM_SESSIONS: dict[str, dict] = {}
_MAX_CUSTOM_SESSIONS = 20


def _value(value: Any):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 4)
    return value


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    data, _ = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    if "completion_date" in data:
        data["completion_date"] = pd.to_datetime(data["completion_date"], errors="coerce")
    return data


def available_data_years() -> list[dict]:
    # The catalog is needed before the user starts a retrain. Reading the
    # identity-resolved trajectory index avoids rebuilding all lifecycle
    # features just to populate the year selectors on page load.
    if TRAJECTORIES.exists():
        indexed = pd.read_csv(
            TRAJECTORIES,
            usecols=["completion_year", "canonical_project_id", "identity_verified"],
            low_memory=False,
        )
        indexed["completion_year"] = pd.to_numeric(indexed["completion_year"], errors="coerce")
        verified = indexed[indexed["identity_verified"].fillna(False)].dropna(
            subset=["completion_year", "canonical_project_id"]
        ).drop_duplicates("canonical_project_id")
        counts = verified.groupby("completion_year").size().sort_index()
        return [{"year": int(year), "completed_projects": int(count)} for year, count in counts.items()]

    data = _dataset()
    unique_projects = data.dropna(subset=["completion_year", "canonical_project_id"]).drop_duplicates("canonical_project_id")
    counts = unique_projects.groupby("completion_year").size().sort_index()
    return [{"year": int(year), "completed_projects": int(count)} for year, count in counts.items()]


def _artifact_bundle(start_year: int, end_year: int, expected_run_id: str | None = None) -> dict:
    target = MODEL_ROOT / f"{start_year}_{end_year}"
    metadata_path = target / "metadata.json"
    required = [metadata_path, target / "cost_model.pkl", target / "delay_model.pkl", target / "risk_model.pkl"]
    if not all(path.exists() for path in required):
        retrained = retrain_lifecycle(start_year, end_year)
        if expected_run_id and retrained.get("run_id") != expected_run_id:
            raise ValueError("Requested lifecycle run is no longer available; retrain to create a new judge session.")
    metadata = json.loads(metadata_path.read_text())
    actual_run_id = metadata.get("run_id")
    if expected_run_id and actual_run_id != expected_run_id:
        raise ValueError(
            "Lifecycle artifacts for this year range were replaced by a different training run. "
            "Retrain and use the newly returned run_id before opening a judge session."
        )
    return {
        "metadata": metadata,
        "cost": joblib.load(target / "cost_model.pkl"),
        "delay": joblib.load(target / "delay_model.pkl"),
        "risk": joblib.load(target / "risk_model.pkl"),
    }


def train_custom(start_year: int, end_year: int, run_id: str | None = None) -> dict:
    """Open a judge session using one exact lifecycle training run.

    The normal UI calls /api/models/retrain immediately before this endpoint and
    passes the returned run_id. Direct callers may omit run_id; artifacts are
    trained only when absent, and the resulting metadata identity is returned.
    """
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("Training start year must be less than or equal to training end year.")

    data = _dataset().copy()
    years = data.completion_year.dropna().astype(int)
    if years.empty:
        raise ValueError("No identity-verified lifecycle outcomes are available.")
    min_year = int(years.min())
    max_year = int(years.max())
    if end_year >= max_year:
        raise ValueError(f"Training must end before {max_year} so at least one later held-out year remains.")
    if end_year < min_year or start_year > max_year:
        raise ValueError(f"Training range must overlap lifecycle data ({min_year}-{max_year}).")

    bundle = _artifact_bundle(start_year, end_year, run_id)
    metadata = bundle["metadata"]
    artifact_run_id = metadata.get("run_id")
    dataset_fingerprint = metadata.get("dataset_fingerprint")
    features = list(metadata["features_used"])

    train_projects = set(data.loc[data.completion_year.between(start_year, end_year), "canonical_project_id"].dropna())
    held_all = data[data.completion_year.gt(end_year)].copy()
    overlap = train_projects & set(held_all.canonical_project_id.dropna())
    if overlap:
        raise ValueError(f"Project-group leakage across judge simulation split: {len(overlap)} project(s)")
    if held_all.empty:
        raise ValueError("No later lifecycle projects are available for a leakage-free historical test.")

    # The judge chooses a project, not an arbitrary repeated monthly row. Use
    # the latest official pre-completion snapshot for that project's displayed
    # prediction while keeping every earlier snapshot available for audit counts.
    held_all = held_all.sort_values(["canonical_project_id", "snapshot_date"])
    held_latest = held_all.drop_duplicates("canonical_project_id", keep="last").reset_index(drop=True)
    held_latest["record_index"] = np.arange(len(held_latest), dtype=int)
    history_counts = held_all.groupby("canonical_project_id").size().to_dict()

    session_id = uuid.uuid4().hex[:16]
    _CUSTOM_SESSIONS[session_id] = {
        "training_start": start_year,
        "training_end": end_year,
        "run_id": artifact_run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "features": features,
        "models": bundle,
        "held_out": held_latest,
        "history_counts": history_counts,
        "predictions": {},
    }
    while len(_CUSTOM_SESSIONS) > _MAX_CUSTOM_SESSIONS:
        _CUSTOM_SESSIONS.pop(next(iter(_CUSTOM_SESSIONS)), None)

    year_counts = held_latest.groupby("completion_year").size().sort_index()
    return {
        "session_id": session_id,
        "run_id": artifact_run_id,
        "dataset_fingerprint": dataset_fingerprint,
        "model_version": metadata["model_version"],
        "model_family": "monthly_lifecycle",
        "training_start": start_year,
        "training_end": end_year,
        "training_samples": int(metadata["training_snapshots"]),
        "training_projects": int(metadata["unique_training_projects"]),
        "features_used": features,
        "feature_count": len(features),
        "selected_algorithms": {**metadata.get("selected_algorithms", {}), "risk": "random_forest"},
        "data_source": "Official PAIMANA/MoSPI monthly lifecycle snapshots",
        "eligible_test_years": [{"year": int(year), "projects": int(count)} for year, count in year_counts.items()],
        "leakage_guard": f"Only lifecycle projects completed in {start_year}-{end_year} were fitted; all offered judge projects complete after {end_year} and are excluded from fitting.",
        "actual_outcomes_sent_to_browser": False,
    }


def _session(session_id: str) -> dict:
    if session_id not in _CUSTOM_SESSIONS:
        raise KeyError(session_id)
    return _CUSTOM_SESSIONS[session_id]


def custom_projects(session_id: str, year: int) -> dict:
    session = _session(session_id)
    if int(year) <= session["training_end"]:
        raise ValueError(f"Test year must be after the training cutoff ({session['training_end']}).")
    rows = session["held_out"][session["held_out"].completion_year.eq(int(year))]
    if rows.empty:
        raise ValueError(f"No held-out lifecycle projects are available for {year}.")
    items = []
    for _, row in rows.iterrows():
        items.append({
            "record_index": int(row.record_index),
            "project_id": _value(row.get("project_id")) or _value(row.get("canonical_project_id")) or "Not published",
            "project_name": _value(row.get("project_name")),
            "sector": _value(row.get("sector")),
            "implementing_agency": _value(row.get("implementing_agency")),
            "approved_cost_cr": _value(row.get("approved_cost_cr")),
            "snapshot_date": _value(row.get("snapshot_date")),
            "held_out_completion_year": int(row.completion_year),
        })
    return {
        "session_id": session_id,
        "run_id": session.get("run_id"),
        "dataset_fingerprint": session.get("dataset_fingerprint"),
        "year": int(year),
        "items": items,
        "actual_outcomes_sent_to_browser": False,
        "note": "Each prediction uses that project's latest official pre-completion lifecycle snapshot; final outcomes remain server-side until reveal.",
    }


def _session_row(session: dict, record_index: int) -> pd.Series:
    rows = session["held_out"][session["held_out"].record_index.eq(int(record_index))]
    if rows.empty:
        raise ValueError("Selected held-out project does not exist in this lifecycle training session.")
    return rows.iloc[0]


def predict_custom(session_id: str, record_index: int) -> dict:
    session = _session(session_id)
    row = _session_row(session, record_index)
    features = session["features"]
    X = row.to_frame().T[features]
    cost_model = session["models"]["cost"]
    delay_model = session["models"]["delay"]
    risk_model = session["models"]["risk"]

    predicted_cost = float(cost_model.predict(X)[0])
    predicted_delay = max(0.0, float(delay_model.predict(X)[0]))
    predicted_risk = str(risk_model.predict(X)[0])
    probability = 1.0
    if hasattr(risk_model, "predict_proba"):
        probability = float(np.asarray(risk_model.predict_proba(X), dtype=float)[0].max())

    project_id = _value(row.get("project_id")) or _value(row.get("canonical_project_id")) or "Not published"
    inputs = {name: _value(row.get(name)) for name in features}
    factors = _shap_factors_for_model(cost_model, row, features)
    prediction = {
        "predicted_cost_overrun": round(predicted_cost, 4),
        "predicted_delay_days": round(predicted_delay, 4),
        "predicted_risk": predicted_risk,
        "risk_probability_percentage": round(probability * 100, 1),
    }
    session["predictions"][int(record_index)] = prediction
    return {
        "session_id": session_id,
        "run_id": session.get("run_id"),
        "dataset_fingerprint": session.get("dataset_fingerprint"),
        "record_index": int(record_index),
        "project": {"project_id": project_id, "project_name": _value(row.get("project_name")), "sector": _value(row.get("sector"))},
        "snapshot_date": _value(row.get("snapshot_date")),
        "history_snapshots": int(session["history_counts"].get(row.get("canonical_project_id"), 1)),
        **prediction,
        "model_inputs": inputs,
        "shap_explanation": factors,
        "confidence_calibration_status": "not_calibrated_for_live_lifecycle_retrain",
        "model_confidence_percentage": None,
        "audit": {
            "model_family": "monthly_lifecycle",
            "run_id": session.get("run_id"),
            "dataset_fingerprint": session.get("dataset_fingerprint"),
            "project_excluded_from_training": True,
            "actual_outcomes_sent_to_browser": False,
            "training_end_year": session["training_end"],
        },
    }


def reveal_custom(session_id: str, record_index: int) -> dict:
    session = _session(session_id)
    row = _session_row(session, record_index)
    prediction = session["predictions"].get(int(record_index))
    if prediction is None:
        raise ValueError("Generate the lifecycle prediction before revealing the actual outcome.")
    actual_cost = float(row.actual_cost_overrun_percentage)
    actual_delay = float(row.actual_delay_days)
    return {
        "session_id": session_id,
        "run_id": session.get("run_id"),
        "dataset_fingerprint": session.get("dataset_fingerprint"),
        "record_index": int(record_index),
        "actual_cost_overrun": round(actual_cost, 4),
        "actual_delay_days": round(actual_delay, 4),
        "actual_risk": str(row.actual_risk),
        "cost_error_absolute_pp": round(abs(prediction["predicted_cost_overrun"] - actual_cost), 4),
        "delay_error_absolute_days": round(abs(prediction["predicted_delay_days"] - actual_delay), 4),
        "completion_date": _value(row.get("completion_date")),
        "source_url": _value(row.get("source_url")) or "",
        "reveal_policy": "Official eventual outcomes were withheld from the browser and model inputs until this explicit reveal request.",
        "actual_outcomes_sent_to_browser": True,
    }
