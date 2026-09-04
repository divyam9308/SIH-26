"""Saved views backed only by frozen production evaluation ledgers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np

from backend.app.ml.provenance import file_sha256
from backend.app.services.frozen_explanation_service import local_explanation
from backend.app.services.operational_driver_service import operational_drivers

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "monthly_lifecycle"
SAVED_WINDOW_ROOT = ROOT / "data" / "processed" / "portfolio_windows"
TRAJECTORIES = ROOT / "data" / "processed" / "paimana_project_trajectories.csv"
RANGE_WINDOWS = {"2001_2017": (2001, 2017, 2018), "2001_2021": (2001, 2021, 2022), "2001_2022": (2001, 2022, 2023)}
# This selector is deliberately lifecycle-only.  ``models/2001_2022`` is a
# historical completed-project CatBoost artifact and is not scientifically
# comparable with the current production lifecycle stack.
CANONICAL_LIFECYCLE_ONLY_WINDOWS = frozenset({"2001_2022"})


def supported_windows() -> list[dict]:
    return [{"key": key, "training_start": start, "training_end": end, "project_start": test, "project_end": 2025, "label": f"{start}-{end} → projects {test}-2025"} for key, (start, end, test) in RANGE_WINDOWS.items()]


def _safe(value, digits=2):
    return None if pd.isna(value) else round(float(value), digits)


def _paths(window: str):
    if window not in RANGE_WINDOWS:
        raise ValueError(f"Unsupported historical window: {window}")
    root = MODEL_ROOT / window
    return root / "run_manifest.json", root / "prediction_validation.csv"


def _saved_window_path(window: str) -> Path:
    if window not in RANGE_WINDOWS:
        raise ValueError(f"Unsupported historical window: {window}")
    return SAVED_WINDOW_ROOT / f"{window}.json"


def _saved_payload(window: str) -> dict | None:
    """Load an existing local frozen project view when its model ledger is absent.

    These files are generated alongside the production artifacts and preserve
    literal per-project prediction, actual, and error values. They are a
    fallback for a missing local model bundle, never a substitute for training.
    """
    if window in CANONICAL_LIFECYCLE_ONLY_WINDOWS:
        return None
    path = _saved_window_path(window)
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text())
        payload = saved["payload"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Saved production view for {window} is malformed: {path}") from exc
    if payload.get("window") != window or not isinstance(payload.get("items"), list):
        raise ValueError(f"Saved production view for {window} has an invalid payload: {path}")
    required = {"project_code", "project_name", "predicted_cost_overrun_percentage", "predicted_delay_days", "risk_level"}
    if payload["items"] and required - set(payload["items"][0]):
        raise ValueError(f"Saved production view for {window} is missing project prediction fields: {path}")
    result = dict(payload)
    result["manifest"] = {
        "model_version": result["items"][0].get("model_version") if result["items"] else window,
        "artifact_source": str(path),
    }
    return result


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
    if manifest_path.exists() and ledger_path.exists():
        manifest = _manifest(window)
        return "|".join(("frozen-ledger-v1", manifest["run_id"], manifest["dataset_fingerprint"], str(manifest_path.stat().st_mtime_ns), str(ledger_path.stat().st_mtime_ns), str(ledger_path.stat().st_size)))
    if window in CANONICAL_LIFECYCLE_ONLY_WINDOWS:
        _manifest(window)
        raise AssertionError("unreachable")
    saved_path = _saved_window_path(window)
    if saved_path.exists():
        stat = saved_path.stat()
        return f"saved-production-view-v1|{saved_path}:{stat.st_mtime_ns}:{stat.st_size}"
    _manifest(window)
    raise AssertionError("unreachable")


@lru_cache(maxsize=1)
def _context() -> pd.DataFrame:
    source = pd.read_csv(TRAJECTORIES, low_memory=False, dtype={"project_id": str})
    source["snapshot_key"] = pd.to_datetime(source["snapshot_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fields = ["project_id", "snapshot_key", "sector", "ministry", "implementing_agency", "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "physical_progress", "financial_progress", "planned_completion_date", "revised_completion_date", "actual_completion_date", "expected_progress_percentage"]
    return source[[field for field in fields if field in source]].drop_duplicates(["project_id", "snapshot_key"], keep="last")


def _score(level: str) -> float:
    # Ordinal severity for UI sorting; it is explicitly not a probability.
    return {"LOW": 25., "MEDIUM": 50., "HIGH": 75., "CRITICAL": 100.}.get(level, 0.)


def _calibration_signature() -> str:
    stat = TRAJECTORIES.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@lru_cache(maxsize=3)
def _delay_severity_calibration(window: str, source_signature: str) -> dict:
    """Fit a severity scale solely from completed projects before the holdout.

    This intentionally does not use holdout values.  It is a decision-support
    policy layered over the frozen Cost/Delay predictions, not a retrained risk
    classifier and not a probability estimate.
    """
    del source_signature
    _, end_year, _ = RANGE_WINDOWS[window]
    columns = ["canonical_project_id", "completion_year", "actual_delay_days", "snapshot_date"]
    history = pd.read_csv(TRAJECTORIES, usecols=lambda name: name in columns, dtype={"canonical_project_id": str}, low_memory=False)
    history["completion_year"] = pd.to_numeric(history["completion_year"], errors="coerce")
    history["actual_delay_days"] = pd.to_numeric(history["actual_delay_days"], errors="coerce")
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], errors="coerce")
    training = history[history["completion_year"].le(end_year)].dropna(subset=["canonical_project_id", "actual_delay_days"])
    training = training.sort_values("snapshot_date").drop_duplicates("canonical_project_id", keep="last")
    values = np.sort(training["actual_delay_days"].clip(lower=0).to_numpy(dtype=float))
    if len(values) < 20:
        raise ValueError(f"Historical severity calibration for {window} has insufficient pre-holdout outcomes.")
    cutoffs = np.quantile(values, [0.25, 0.50, 0.75]).astype(float)
    return {
        "training_end_year": end_year,
        "training_projects": int(len(values)),
        "values": values,
        "cutoffs": cutoffs,
        "method": "pre_holdout_actual_delay_empirical_cdf_v1",
    }


def _calibrated_delay_severity(predicted_delay_days: float | None, calibration: dict) -> tuple[float, str]:
    if predicted_delay_days is None or not np.isfinite(predicted_delay_days):
        return 0.0, "LOW"
    delay = max(0.0, float(predicted_delay_days))
    score = round(float(np.searchsorted(calibration["values"], delay, side="right") / len(calibration["values"]) * 100), 1)
    low, medium, high = calibration["cutoffs"]
    if delay <= low:
        level = "LOW"
    elif delay <= medium:
        level = "MEDIUM"
    elif delay <= high:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return score, level


@lru_cache(maxsize=3)
def _payload(window: str, signature: str) -> dict:
    del signature
    saved_payload = _saved_payload(window)
    manifest_path, ledger_path = _paths(window)
    if not manifest_path.exists() or not ledger_path.exists():
        if saved_payload is not None:
            return saved_payload
        _manifest(window)
        raise AssertionError("unreachable")
    manifest = _manifest(window)
    calibration = _delay_severity_calibration(window, _calibration_signature())
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
        model_level = str(row["predicted_risk"]).upper() if pd.notna(row["predicted_risk"]) else "LOW"
        severity_score, level = _calibrated_delay_severity(delay, calibration)
        items.append({
            "project_code": str(row["canonical_project_id"]), "project_name": str(row["project_name"]), "sector": str(row.get("sector") or "Not reported"), "ministry": None if pd.isna(row.get("ministry")) else str(row["ministry"]), "implementing_agency": None if pd.isna(row.get("implementing_agency")) else str(row["implementing_agency"]), "snapshot_date": str(row["snapshot_date"]),
            "original_cost_cr": approved, "revised_cost_cr": _safe(row.get("revised_cost_cr")), "expenditure_cr": _safe(row.get("cumulative_expenditure_cr")), "physical_progress_pct": _safe(row.get("physical_progress"), 1),
            "predicted_cost_overrun_percentage": cost, "predicted_cost_overrun_amount_cr": None if cost is None or approved is None else round(approved * cost / 100, 2), "predicted_final_cost_cr": None if cost is None or approved is None else round(approved * (1 + cost / 100), 2), "predicted_delay_days": delay, "predicted_delay_months": None if delay is None else round(delay / 30.4375, 1), "predicted_completion_date": None,
            "actual_cost_overrun_percentage": _safe(row["actual_cost_overrun_percentage"]), "actual_delay_days": _safe(row["actual_delay_days"], 1), "cost_error_percentage": _safe(abs(float(row["cost_error"])) if pd.notna(row["cost_error"]) else None), "delay_error_days": _safe(abs(float(row["delay_error"])) if pd.notna(row["delay_error"]) else None, 1),
            "risk_score": severity_score, "risk_probability_percentage": None, "risk_level": level, "model_version": manifest["model_version"], "model_scope": "Frozen production evaluation ledger; calibrated delay-severity policy fitted only on pre-holdout completed projects. This score is not a probability.", "inference_timestamp": timestamp, "model_confidence_percentage": None, "confidence_calibration_status": "Unavailable: no calibrated per-row confidence in the frozen ledger.", "schedule_risk_probability": None, "cost_risk_probability": None, "estimated_schedule_extension_days": delay, "estimated_cost_escalation_pct": cost, "priority_score": severity_score, "priority_level": level.lower(), "confidence": "production holdout", "exposure_percentile": None, "best_models": {"schedule_regressor": manifest["model_version"], "cost_regressor": manifest["model_version"]}, "schedule_drivers": [], "cost_drivers": [], "observed": {"schedule_extension_days": _safe(row["actual_delay_days"], 1), "cost_escalation_pct": _safe(row["actual_cost_overrun_percentage"]), "financial_progress_pct": None, "physical_progress_pct": _safe(row.get("physical_progress"), 1)}})
    return {"items": items, "generated_at": timestamp, "window": window, "manifest": manifest, "risk_policy": {"method": calibration["method"], "training_end_year": calibration["training_end_year"], "training_projects": calibration["training_projects"], "delay_cutoffs_days": [round(float(value), 2) for value in calibration["cutoffs"]]}}


def portfolio_payload(window: str) -> dict:
    return _payload(window, _signature(window))


def write_saved_window_view(window: str) -> Path:
    """Write an auditable convenience view derived from the canonical ledger.

    The API continues to read the lifecycle manifest and ledger directly.  The
    JSON is for frontend/offline consumers and carries the exact source hashes
    so it can never masquerade as an independent model artifact.
    """
    payload = portfolio_payload(window)
    manifest_path, ledger_path = _paths(window)
    destination = _saved_window_path(window)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "source": "canonical_monthly_lifecycle_evaluation_ledger",
        "manifest_sha256": file_sha256(manifest_path),
        "prediction_ledger_sha256": file_sha256(ledger_path),
        "payload": payload,
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return destination


def historical_project(code: str, window: str) -> dict:
    matches = [row for row in portfolio_payload(window)["items"] if row["project_code"] == str(code)]
    if not matches:
        raise KeyError(code)
    item = max(matches, key=lambda row: row["snapshot_date"])
    record = {"snapshot_date": item["snapshot_date"], "sector": item["sector"], "ministry": item["ministry"], "implementing_agency": item["implementing_agency"], "project_code": item["project_code"], "project_name": item["project_name"], "original_cost_cr": item["original_cost_cr"], "revised_cost_cr": item["revised_cost_cr"], "expenditure_cr": item["expenditure_cr"], "original_end_date": None, "revised_end_date": None, "physical_progress_pct": item["physical_progress_pct"], "source_url": "", "days_to_original_deadline": 0, "schedule_extension_days": item["actual_delay_days"], "cost_escalation_pct": item["actual_cost_overrun_percentage"], "expenditure_to_original_pct": None, "financial_progress_pct": item.get("financial_progress_pct"), "schedule_overrun_90d": None, "cost_overrun_5pct": None, "dq_expenditure_gt_revised": 0, "dq_revised_date_before_original": 0, "dq_missing_revised_cost": int(item["revised_cost_cr"] is None), "dq_missing_revised_date": 1, "dq_missing_progress": int(item["physical_progress_pct"] is None)}
    best_models = item["best_models"]
    explanation_status = {
        "available": False,
        "reason": "Project-level SHAP was not persisted for this frozen evaluation run.",
        "source": "frozen_evaluation_ledger",
    }
    stored_explanation = local_explanation(window, item["project_code"], item["snapshot_date"])
    if stored_explanation:
        def stored(target: str):
            return stored_explanation["models"][target]["factors"]
        available_status = {"available": True, "reason": None, "source": "verified_frozen_local_shap_ledger"}
        cost_factors, delay_factors, risk_factors = stored("cost"), stored("delay"), stored("risk")
    else:
        cost_factors = delay_factors = risk_factors = []
        available_status = explanation_status
    target = pd.to_datetime(item["snapshot_date"], errors="coerce")
    history = _trajectory_history()
    history = history[(history["canonical_project_id"].eq(str(code))) & (history["snapshot_date"].le(target))]
    current = history.sort_values("snapshot_date").drop_duplicates("snapshot_date", keep="last").tail(1)
    operational = operational_drivers(current.iloc[0], history, source="official_snapshot_trajectory") if not current.empty else []
    forecast = {"project_id": item["project_code"], "project_name": item["project_name"], "model_version": item["model_version"], "dataset_snapshot_date": item["snapshot_date"], "inference_timestamp": item["inference_timestamp"], "current_status": {"snapshot_month": item["snapshot_date"], "physical_progress_percentage": item["physical_progress_pct"], "current_estimated_cost": item["revised_cost_cr"], "expenditure_cr": item["expenditure_cr"], "planned_completion_date": None, "progress_delay_percentage_points": None}, "predicted_cost_overrun_percentage": item["predicted_cost_overrun_percentage"], "predicted_cost_overrun_amount_cr": item["predicted_cost_overrun_amount_cr"], "predicted_final_cost_cr": item["predicted_final_cost_cr"], "predicted_delay_days": item["predicted_delay_days"], "predicted_cost_overrun": item["predicted_cost_overrun_percentage"], "predicted_completion_date": None, "current_progress": item["physical_progress_pct"], "predicted_delay_months": item["predicted_delay_months"], "risk_score": item["risk_score"], "risk_probability_percentage": None, "risk_level": item["risk_level"], "model_confidence_percentage": None, "confidence_calibration_status": item["confidence_calibration_status"], "explanation": cost_factors, "shap_explanation": cost_factors, "cost_factors": cost_factors, "delay_factors": delay_factors, "risk_factors": risk_factors, "cost_explanation_status": available_status, "delay_explanation_status": available_status, "risk_explanation_status": available_status, "operational_drivers": operational, "best_models": {"cost": best_models.get("cost_regressor") or best_models.get("cost") or item["model_version"], "delay": best_models.get("schedule_regressor") or best_models.get("delay") or item["model_version"]}, "expected_range": None, "completion_probabilities": [], "features_used": [], "model_scope": item["model_scope"]}
    return {"record": record, "forecast": forecast}


def historical_peer_benchmark(code: str, window: str, limit: int = 6) -> dict:
    """Comparable projects from the same frozen evaluation ledger, never live data."""
    item = historical_project(code, window)["record"]
    frame = pd.DataFrame(portfolio_payload(window)["items"])
    peers = frame[frame["project_code"].ne(str(code))].copy()
    same_sector = peers[peers["sector"].eq(item["sector"])]
    if not same_sector.empty:
        peers = same_sector
    base = item["original_cost_cr"]
    if base is not None:
        peers["cost_distance"] = (np.log1p(pd.to_numeric(peers["original_cost_cr"], errors="coerce")) - np.log1p(float(base))).abs()
        peers = peers.sort_values("cost_distance", na_position="last")
    peers = peers.head(limit)

    def median(name: str):
        values = pd.to_numeric(peers[name], errors="coerce").dropna()
        return None if values.empty else round(float(values.median()), 2)

    return {
        "sector": item["sector"], "peer_count": int(len(peers)),
        "medians": {"original_cost_cr": median("original_cost_cr"), "cost_escalation_pct": median("actual_cost_overrun_percentage"), "schedule_extension_days": median("actual_delay_days"), "financial_progress_pct": None, "physical_progress_pct": median("physical_progress_pct")},
        "peers": [{"project_code": str(row.project_code), "project_name": str(row.project_name), "original_cost_cr": _safe(row.original_cost_cr), "cost_escalation_pct": _safe(row.actual_cost_overrun_percentage), "schedule_extension_days": _safe(row.actual_delay_days, 1)} for row in peers.itertuples()],
    }


@lru_cache(maxsize=1)
def _trajectory_history() -> pd.DataFrame:
    columns = ["canonical_project_id", "snapshot_date", "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "physical_progress", "financial_progress", "expected_progress_percentage", "planned_completion_date", "revised_completion_date", "actual_completion_date"]
    frame = pd.read_csv(TRAJECTORIES, usecols=lambda name: name in columns, dtype={"canonical_project_id": str}, low_memory=False)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    for field in ("planned_completion_date", "revised_completion_date", "actual_completion_date"):
        frame[field] = pd.to_datetime(frame[field], errors="coerce")
    return frame


def historical_warnings(code: str, window: str) -> dict:
    """Snapshot-change events at the same frozen-ledger snapshot as project detail."""
    item = historical_project(code, window)["record"]
    target = pd.to_datetime(item["snapshot_date"], errors="coerce")
    history = _trajectory_history()
    rows = history[(history["canonical_project_id"].eq(str(code))) & (history["snapshot_date"].le(target))].dropna(subset=["snapshot_date"]).sort_values("snapshot_date").drop_duplicates("snapshot_date", keep="last")
    if len(rows) < 2:
        return {"available": False, "reason": "Historical warning-event history was not persisted for this evaluation window.", "source": "official_snapshot_trajectory", "items": []}
    previous, current = rows.iloc[-2], rows.iloc[-1]
    events: list[dict] = []
    date = current.snapshot_date.strftime("%Y-%m-%d")
    def number(value): return None if pd.isna(value) else float(value)
    old_cost, new_cost = number(previous.revised_cost_cr), number(current.revised_cost_cr)
    if old_cost is not None and new_cost is not None and new_cost > old_cost:
        pct = (new_cost - old_cost) / max(old_cost, 1.0) * 100
        events.append({"date": date, "type": "revised_cost_increase", "severity": "HIGH" if pct >= 10 else "MEDIUM", "message": f"Revised cost increased by {pct:.1f}% from the previous official snapshot."})
    old_end, new_end = previous.revised_completion_date, current.revised_completion_date
    if pd.notna(old_end) and pd.notna(new_end) and new_end > old_end:
        days = int((new_end - old_end).days)
        events.append({"date": date, "type": "completion_date_extended", "severity": "HIGH" if days >= 90 else "MEDIUM", "message": f"Revised completion date moved later by {days} days from the previous official snapshot."})
    old_spend, new_spend = number(previous.cumulative_expenditure_cr), number(current.cumulative_expenditure_cr)
    old_progress, new_progress = number(previous.physical_progress), number(current.physical_progress)
    if old_spend is not None and new_spend is not None and new_spend > old_spend and old_progress is not None and new_progress is not None and new_progress <= old_progress:
        events.append({"date": date, "type": "spend_without_physical_progress", "severity": "MEDIUM", "message": "Cumulative expenditure increased while reported physical progress did not increase."})
    if pd.notna(current.planned_completion_date) and previous.snapshot_date < current.planned_completion_date <= current.snapshot_date and pd.isna(current.actual_completion_date):
        events.append({"date": date, "type": "planned_deadline_crossed", "severity": "HIGH", "message": "The original planned completion date was crossed without a reported actual completion date."})
    return {"available": True, "reason": None, "source": "official_snapshot_trajectory", "items": events}


def invalidate_range_cache() -> None:
    _payload.cache_clear()
    _context.cache_clear()
    _delay_severity_calibration.cache_clear()
