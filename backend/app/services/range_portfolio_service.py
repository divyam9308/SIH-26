"""Saved views backed only by frozen production evaluation ledgers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "monthly_lifecycle"
TRAJECTORIES = ROOT / "data" / "processed" / "paimana_project_trajectories.csv"
RANGE_WINDOWS = {"2001_2017": (2001, 2017, 2018), "2001_2021": (2001, 2021, 2022), "2001_2022": (2001, 2022, 2023)}


def supported_windows() -> list[dict]:
    return [{"key": key, "training_start": start, "training_end": end, "project_start": test, "project_end": 2025, "label": f"{start}-{end} → projects {test}-2025"} for key, (start, end, test) in RANGE_WINDOWS.items()]


def _safe(value, digits=2):
    return None if pd.isna(value) else round(float(value), digits)


def _paths(window: str):
    if window not in RANGE_WINDOWS:
        raise ValueError(f"Unsupported historical window: {window}")
    root = MODEL_ROOT / window
    return root / "run_manifest.json", root / "prediction_validation.csv"


def _manifest(window: str) -> dict:
    manifest_path, ledger_path = _paths(window)
    if not manifest_path.exists() or not ledger_path.exists():
        raise ValueError(f"Production evaluation for {window} is unavailable; its frozen bundle has not been published.")
    value = json.loads(manifest_path.read_text())
    if value.get("status") != "complete" or value.get("model_role") != "production":
        raise ValueError(f"Production evaluation for {window} is unavailable; its bundle is incomplete.")
    if not all(value.get(field) for field in ("model_version", "run_id", "dataset_fingerprint")):
        raise ValueError(f"Production evaluation for {window} is unavailable; its provenance is incomplete.")
    return value


def _signature(window: str) -> str:
    manifest_path, ledger_path = _paths(window)
    manifest = _manifest(window)
    return "|".join(("frozen-ledger-v1", manifest["run_id"], manifest["dataset_fingerprint"], str(manifest_path.stat().st_mtime_ns), str(ledger_path.stat().st_mtime_ns), str(ledger_path.stat().st_size)))


@lru_cache(maxsize=1)
def _context() -> pd.DataFrame:
    source = pd.read_csv(TRAJECTORIES, low_memory=False, dtype={"project_id": str})
    source["snapshot_key"] = pd.to_datetime(source["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fields = ["project_id", "snapshot_key", "sector", "ministry", "implementing_agency", "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "physical_progress"]
    return source[[field for field in fields if field in source]].drop_duplicates(["project_id", "snapshot_key"], keep="last")


def _score(level: str) -> float:
    # Ordinal severity for UI sorting; it is explicitly not a probability.
    return {"LOW": 25., "MEDIUM": 50., "HIGH": 75., "CRITICAL": 100.}.get(level, 0.)


@lru_cache(maxsize=3)
def _payload(window: str, signature: str) -> dict:
    del signature
    manifest = _manifest(window)
    _, ledger_path = _paths(window)
    ledger = pd.read_csv(ledger_path, low_memory=False, dtype={"canonical_project_id": str})
    required = {"canonical_project_id", "project_name", "snapshot_date", "predicted_cost_overrun", "predicted_delay_days", "predicted_risk", "actual_cost_overrun_percentage", "actual_delay_days", "cost_error", "delay_error"}
    if missing := required - set(ledger):
        raise ValueError(f"Production evaluation ledger for {window} is missing: {', '.join(sorted(missing))}.")
    ledger["snapshot_key"] = pd.to_datetime(ledger["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = ledger.merge(_context(), how="left", left_on=["canonical_project_id", "snapshot_key"], right_on=["project_id", "snapshot_key"])
    # The ledger is snapshot-level because that is how temporal production MAE
    # is evaluated. The UI is a project register, so use each project's latest
    # official holdout snapshot; its predictions and errors remain literal rows
    # from the frozen ledger, while the manifest retains headline all-row MAE.
    frame["_snapshot_sort"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.sort_values("_snapshot_sort").drop_duplicates("canonical_project_id", keep="last")
    timestamp = datetime.now(timezone.utc).isoformat()
    items = []
    for _, row in frame.iterrows():
        cost, delay, approved = _safe(row["predicted_cost_overrun"]), _safe(row["predicted_delay_days"], 1), _safe(row.get("approved_cost_cr"))
        level = str(row["predicted_risk"]).upper() if pd.notna(row["predicted_risk"]) else "LOW"
        items.append({
            "project_code": str(row["canonical_project_id"]), "project_name": str(row["project_name"]), "sector": str(row.get("sector") or "Not reported"), "ministry": None if pd.isna(row.get("ministry")) else str(row["ministry"]), "implementing_agency": None if pd.isna(row.get("implementing_agency")) else str(row["implementing_agency"]), "snapshot_date": str(row["snapshot_date"]),
            "original_cost_cr": approved, "revised_cost_cr": _safe(row.get("revised_cost_cr")), "expenditure_cr": _safe(row.get("cumulative_expenditure_cr")), "physical_progress_pct": _safe(row.get("physical_progress"), 1),
            "predicted_cost_overrun_percentage": cost, "predicted_cost_overrun_amount_cr": None if cost is None or approved is None else round(approved * cost / 100, 2), "predicted_final_cost_cr": None if cost is None or approved is None else round(approved * (1 + cost / 100), 2), "predicted_delay_days": delay, "predicted_delay_months": None if delay is None else round(delay / 30.4375, 1), "predicted_completion_date": None,
            "actual_cost_overrun_percentage": _safe(row["actual_cost_overrun_percentage"]), "actual_delay_days": _safe(row["actual_delay_days"], 1), "cost_error_percentage": _safe(abs(float(row["cost_error"])) if pd.notna(row["cost_error"]) else None), "delay_error_days": _safe(abs(float(row["delay_error"])) if pd.notna(row["delay_error"]) else None, 1),
            "risk_score": _score(level), "risk_probability_percentage": None, "risk_level": level, "model_version": manifest["model_version"], "model_scope": "Frozen production evaluation ledger; one row per official holdout snapshot.", "inference_timestamp": timestamp, "model_confidence_percentage": None, "confidence_calibration_status": "Unavailable: no calibrated per-row confidence in the frozen ledger.", "schedule_risk_probability": None, "cost_risk_probability": None, "estimated_schedule_extension_days": delay, "estimated_cost_escalation_pct": cost, "priority_score": _score(level), "priority_level": level.lower(), "confidence": "production holdout", "exposure_percentile": None, "best_models": {"schedule_regressor": manifest["model_version"], "cost_regressor": manifest["model_version"]}, "schedule_drivers": [], "cost_drivers": [], "observed": {"schedule_extension_days": _safe(row["actual_delay_days"], 1), "cost_escalation_pct": _safe(row["actual_cost_overrun_percentage"]), "financial_progress_pct": None, "physical_progress_pct": _safe(row.get("physical_progress"), 1)}})
    return {"items": items, "generated_at": timestamp, "window": window, "manifest": manifest}


def portfolio_payload(window: str) -> dict:
    return _payload(window, _signature(window))


def historical_project(code: str, window: str) -> dict:
    matches = [row for row in portfolio_payload(window)["items"] if row["project_code"] == str(code)]
    if not matches:
        raise KeyError(code)
    item = max(matches, key=lambda row: row["snapshot_date"])
    record = {"snapshot_date": item["snapshot_date"], "sector": item["sector"], "ministry": item["ministry"], "implementing_agency": item["implementing_agency"], "project_code": item["project_code"], "project_name": item["project_name"], "original_cost_cr": item["original_cost_cr"], "revised_cost_cr": item["revised_cost_cr"], "expenditure_cr": item["expenditure_cr"], "original_end_date": None, "revised_end_date": None, "physical_progress_pct": item["physical_progress_pct"], "source_url": "", "days_to_original_deadline": 0, "schedule_extension_days": item["actual_delay_days"], "cost_escalation_pct": item["actual_cost_overrun_percentage"], "expenditure_to_original_pct": None, "financial_progress_pct": None, "schedule_overrun_90d": None, "cost_overrun_5pct": None, "dq_expenditure_gt_revised": 0, "dq_revised_date_before_original": 0, "dq_missing_revised_cost": int(item["revised_cost_cr"] is None), "dq_missing_revised_date": 1, "dq_missing_progress": int(item["physical_progress_pct"] is None)}
    forecast = {"project_id": item["project_code"], "project_name": item["project_name"], "model_version": item["model_version"], "dataset_snapshot_date": item["snapshot_date"], "inference_timestamp": item["inference_timestamp"], "current_status": {"snapshot_month": item["snapshot_date"], "physical_progress_percentage": item["physical_progress_pct"], "current_estimated_cost": item["revised_cost_cr"], "expenditure_cr": item["expenditure_cr"], "planned_completion_date": None, "progress_delay_percentage_points": None}, "predicted_cost_overrun_percentage": item["predicted_cost_overrun_percentage"], "predicted_cost_overrun_amount_cr": item["predicted_cost_overrun_amount_cr"], "predicted_final_cost_cr": item["predicted_final_cost_cr"], "predicted_delay_days": item["predicted_delay_days"], "predicted_cost_overrun": item["predicted_cost_overrun_percentage"], "predicted_completion_date": None, "current_progress": item["physical_progress_pct"], "predicted_delay_months": item["predicted_delay_months"], "risk_score": item["risk_score"], "risk_probability_percentage": None, "risk_level": item["risk_level"], "model_confidence_percentage": None, "confidence_calibration_status": item["confidence_calibration_status"], "explanation": [], "shap_explanation": [], "cost_factors": [], "delay_factors": [], "risk_factors": [], "best_models": item["best_models"], "expected_range": None, "completion_probabilities": [], "features_used": [], "model_scope": item["model_scope"]}
    return {"record": record, "forecast": forecast}


def invalidate_range_cache() -> None:
    _payload.cache_clear()
    _context.cache_clear()
