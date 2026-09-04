"""Inference and audit APIs for monthly lifecycle models."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import OUTCOMES, SNAPSHOTS, SNAPSHOTS_GZ, TRAJECTORIES, engineer_as_of_features, load_monthly_snapshots, resolve_identities
from backend.app.ml.production_cost_baseline import enrich_history_for_production, target_feature_contract
from backend.app.ml.production_delay_baseline import DEFAULT_PRODUCTION_WINDOW, enrich_history_for_delay_production
from backend.app.ml.provenance import file_sha256
from backend.app.services.simulation_service import _shap_factors_for_model

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "monthly_lifecycle"
COMPARISON = ROOT / "reports" / "monthly_lifecycle_model_comparison.json"


def lifecycle_comparison() -> dict:
    if not COMPARISON.exists():
        return {"available": False, "reason": "Monthly lifecycle training report has not been generated yet.", "windows": []}
    return {"available": True, **json.loads(COMPARISON.read_text())}


def _current_source_hashes() -> dict[str, str | None]:
    snapshot_path = SNAPSHOTS if SNAPSHOTS.exists() else SNAPSHOTS_GZ
    return {
        "monthly_snapshots": file_sha256(snapshot_path) if snapshot_path.exists() else None,
        "completed_outcomes": file_sha256(OUTCOMES) if OUTCOMES.exists() else None,
    }


def _validate_bundle_provenance(window: str, metadata: dict, manifest: dict) -> None:
    if not manifest:
        return
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Lifecycle model {window} is not provenance-valid ({manifest.get('status') or 'invalid manifest'}). Retrain this window before inference."
        )
    manifest_run = manifest.get("run_id")
    metadata_run = metadata.get("run_id") or (metadata.get("provenance") or {}).get("run_id")
    manifest_dataset = manifest.get("dataset_fingerprint")
    metadata_dataset = metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint")
    if not manifest_run or not metadata_run or not manifest_dataset or not metadata_dataset:
        raise RuntimeError(f"Lifecycle model {window} has an incomplete provenance manifest. Retrain this window before inference.")
    if manifest_run != metadata_run:
        raise RuntimeError(f"Lifecycle model {window} failed provenance validation: manifest/metadata run IDs differ.")
    if manifest_dataset != metadata_dataset:
        raise RuntimeError(f"Lifecycle model {window} failed provenance validation: manifest/metadata dataset fingerprints differ.")

    expected_sources = manifest.get("source_dataset_files") or {}
    current_sources = _current_source_hashes()
    for name, expected in expected_sources.items():
        current = current_sources.get(name)
        if expected and current and expected != current:
            raise RuntimeError(
                f"Lifecycle model {window} was trained against a different {name} dataset. Retrain before inference."
            )


@lru_cache(maxsize=2)
def _bundle(window: str) -> dict:
    target = MODEL_ROOT / window
    if not target.exists():
        raise FileNotFoundError(f"Monthly lifecycle model {window} is not available")
    manifest_path = target / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    metadata = json.loads((target / "metadata.json").read_text())
    _validate_bundle_provenance(window, metadata, manifest)
    return {
        "metadata": metadata,
        "manifest": manifest,
        "importance": json.loads((target / "shap_importance.json").read_text()),
        "cost": joblib.load(target / "cost_model.pkl"),
        "delay": joblib.load(target / "delay_model.pkl"),
        "risk": joblib.load(target / "risk_model.pkl"),
    }


def _inference_source_signature() -> str:
    paths = [TRAJECTORIES] if TRAJECTORIES.exists() else [SNAPSHOTS if SNAPSHOTS.exists() else SNAPSHOTS_GZ, OUTCOMES]
    parts = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


@lru_cache(maxsize=1)
def _inference_frame_cached(source_signature: str) -> pd.DataFrame:
    del source_signature
    if TRAJECTORIES.exists():
        frame = pd.read_csv(TRAJECTORIES, dtype={"project_id": "string"}, low_memory=False)
        frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date, errors="coerce")
        frame = enrich_history_for_production(frame)
        return enrich_history_for_delay_production(frame)
    snapshots = load_monthly_snapshots()
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, _ = resolve_identities(snapshots, outcomes)
    frame = engineer_as_of_features(resolved, outcomes)
    frame = enrich_history_for_production(frame)
    return enrich_history_for_delay_production(frame)


def _inference_frame() -> pd.DataFrame:
    return _inference_frame_cached(_inference_source_signature())


def lifecycle_project_forecast(code: str, window: str = DEFAULT_PRODUCTION_WINDOW) -> dict:
    code = str(code).strip().upper(); frame = _inference_frame(); rows = frame[frame.project_id.astype("string").str.upper().eq(code)].sort_values("snapshot_date")
    if rows.empty:
        raise KeyError(code)
    latest = rows.iloc[-1]; bundle = _bundle(window); feature_contract = target_feature_contract(bundle["metadata"])
    cost_features = feature_contract["cost"]; delay_features = feature_contract["delay"]; risk_features = feature_contract["risk"]
    cost_X = latest.to_frame().T.reindex(columns=cost_features)
    delay_X = latest.to_frame().T.reindex(columns=delay_features)
    risk_X = latest.to_frame().T.reindex(columns=risk_features)
    cost = float(bundle["cost"].predict(cost_X)[0]); delay = max(0.0, float(bundle["delay"].predict(delay_X)[0])); risk = str(bundle["risk"].predict(risk_X)[0])
    importance = bundle["importance"]
    global_factors = [{"feature": item["feature"], "importance": item["importance"]} for item in importance.get("cost", {}).get("features", [])[:8]]
    explain_model = getattr(bundle["cost"], "model", bundle["cost"])
    explain_features = list(getattr(bundle["cost"], "features", cost_features))
    local_factors = _shap_factors_for_model(explain_model, latest, explain_features)
    all_features = list(dict.fromkeys(cost_features + delay_features + risk_features))
    inputs = {name: (None if pd.isna(latest.get(name)) else latest.get(name)) for name in all_features}
    for key, value in list(inputs.items()):
        if isinstance(value, (np.integer, np.floating)):
            inputs[key] = value.item()
        elif isinstance(value, pd.Timestamp):
            inputs[key] = value.strftime("%Y-%m-%d")
    provenance = bundle["metadata"].get("provenance") or {}
    cost_baseline = bundle["metadata"].get("production_cost_baseline")
    delay_baseline = bundle["metadata"].get("production_delay_baseline")
    return {
        "project_id": code,
        "project_name": latest.project_name,
        "model_version": bundle["metadata"]["model_version"],
        "snapshot_date": pd.Timestamp(latest.snapshot_date).strftime("%Y-%m-%d"),
        "history_snapshots": int(len(rows)),
        "predicted_cost_overrun_percentage": round(cost, 2),
        "predicted_delay_days": round(delay, 1),
        "risk_level": risk,
        "model_inputs": inputs,
        "cost_features_used": cost_features,
        "delay_features_used": delay_features,
        "risk_features_used": risk_features,
        "production_cost_baseline": cost_baseline,
        "production_delay_baseline": delay_baseline,
        "promoted_from_experiment": bundle["metadata"].get("promoted_from_experiment"),
        "promoted_delay_from_experiment": bundle["metadata"].get("promoted_delay_from_experiment"),
        "shap_explanation": local_factors,
        "global_feature_importance": global_factors,
        "explanation_scope": "shap_explanation is project-specific for the displayed Cost snapshot; global_feature_importance is aggregate Cost training-sample importance.",
        "provenance": {
            "run_id": bundle["metadata"].get("run_id") or provenance.get("run_id"),
            "dataset_fingerprint": bundle["metadata"].get("dataset_fingerprint") or provenance.get("dataset_fingerprint"),
            "verified": bool(bundle.get("manifest") and bundle["manifest"].get("status") == "complete"),
        },
        "model_scope": f"Official PAIMANA monthly lifecycle production model; Cost baseline={cost_baseline}; Delay baseline={delay_baseline}; Risk retains the existing production contract.",
    }


def forecast_evolution(project_id: str, window: str = DEFAULT_PRODUCTION_WINDOW) -> dict:
    path = MODEL_ROOT / window / "prediction_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validation rows for {window} are unavailable")
    frame = pd.read_csv(path, dtype={"canonical_project_id": "string"})
    rows = frame[frame.canonical_project_id.eq(str(project_id))].sort_values("snapshot_date")
    safe = rows.astype(object).where(pd.notna(rows), None)
    return {"model_version": window, "project_id": project_id, "items": safe.to_dict("records"), "count": int(len(rows)),
            "source_policy": "Every point is an official historical snapshot; no synthetic interpolation is used."}
