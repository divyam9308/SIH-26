from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import math

import numpy as np
import pandas as pd

from backend.app.ml.real_time_windows import active_version
from backend.app.services.data_service import project_dataset_signature, projects_df
from backend.app.services.prediction_service import active_model_signature, project_forecast
from backend.app.services.range_portfolio_service import portfolio_payload as historical_portfolio_payload, supported_windows


def _safe_number(value, digits: int = 2):
    return None if pd.isna(value) else round(float(value), digits)


@lru_cache(maxsize=4)
def _portfolio_payload_cached(model_version: str, model_signature: str, dataset_signature: str) -> dict:
    """Cache non-SHAP portfolio inference against model and dataset identity."""
    del model_signature, dataset_signature
    frame = projects_df().copy()
    forecasts = [project_forecast(str(code), include_explanations=False) for code in frame["project_code"]]
    generated_at = datetime.now(timezone.utc).isoformat()
    exposure_rank = frame["original_cost_cr"].rank(method="average", pct=True).fillna(.4).to_numpy()
    rows: list[dict] = []
    for index, (_, project) in enumerate(frame.iterrows()):
        forecast = forecasts[index]
        delay_days = max(0.0, float(forecast["predicted_delay_days"]))
        cost_pct = float(forecast["predicted_cost_overrun_percentage"])
        schedule_probability = float(np.clip(delay_days / 730, 0, 1))
        cost_probability = float(np.clip(max(0.0, cost_pct) / 40, 0, 1))
        priority = max(0.0, min(100.0, 100 * (.45 * schedule_probability + .35 * cost_probability + .20 * float(exposure_rank[index]))))
        completeness = sum(pd.notna(project.get(column)) for column in ("revised_cost_cr", "expenditure_cr", "physical_progress_pct", "revised_end_date")) / 4
        confidence = "high" if completeness >= .75 else "medium" if completeness >= .5 else "low"
        rows.append({
            "project_code": str(project["project_code"]),
            "project_name": str(project["project_name"]),
            "sector": str(project["sector"]),
            "ministry": None if pd.isna(project.get("ministry")) else str(project.get("ministry")),
            "implementing_agency": None,
            "snapshot_date": pd.Timestamp(project["snapshot_date"]).strftime("%Y-%m-%d"),
            "original_cost_cr": _safe_number(project.get("original_cost_cr")),
            "revised_cost_cr": _safe_number(project.get("revised_cost_cr")),
            "expenditure_cr": _safe_number(project.get("expenditure_cr")),
            "physical_progress_pct": _safe_number(project.get("physical_progress_pct"), 1),
            "predicted_cost_overrun_percentage": forecast["predicted_cost_overrun_percentage"],
            "predicted_cost_overrun_amount_cr": forecast["predicted_cost_overrun_amount_cr"],
            "predicted_final_cost_cr": forecast["predicted_final_cost_cr"],
            "predicted_delay_days": forecast["predicted_delay_days"],
            "predicted_delay_months": forecast["predicted_delay_months"],
            "predicted_completion_date": forecast["predicted_completion_date"],
            "actual_cost_overrun_percentage": None,
            "actual_delay_days": None,
            "cost_error_percentage": None,
            "delay_error_days": None,
            "risk_score": forecast["risk_score"],
            "risk_probability_percentage": forecast["risk_probability_percentage"],
            "risk_level": forecast["risk_level"],
            "model_version": model_version,
            "model_scope": "temporal cost and delay forecasting",
            "inference_timestamp": generated_at,
            "model_confidence_percentage": forecast["model_confidence_percentage"],
            "confidence_calibration_status": forecast["confidence_calibration_status"],
            "schedule_risk_probability": round(schedule_probability, 4),
            "cost_risk_probability": round(cost_probability, 4),
            "estimated_schedule_extension_days": round(delay_days, 1),
            "estimated_cost_escalation_pct": round(cost_pct, 2),
            "priority_score": round(priority, 1),
            "priority_level": str(forecast["risk_level"]).lower(),
            "confidence": confidence,
            "exposure_percentile": round(float(exposure_rank[index]), 4),
            "best_models": {
                "schedule_classifier": forecast["best_models"]["delay"],
                "cost_classifier": forecast["best_models"]["cost"],
                "schedule_regressor": forecast["best_models"]["delay"],
                "cost_regressor": forecast["best_models"]["cost"],
            },
            "schedule_drivers": [],
            "cost_drivers": [],
            "observed": {
                "schedule_extension_days": _safe_number(project.get("schedule_extension_days"), 1),
                "cost_escalation_pct": _safe_number(project.get("cost_escalation_pct")),
                "financial_progress_pct": _safe_number(project.get("financial_progress_pct"), 1),
                "physical_progress_pct": _safe_number(project.get("physical_progress_pct"), 1),
            },
        })
    return {"items": rows, "generated_at": generated_at}


def portfolio_payload(window: str | None = None) -> dict:
    if window:
        return historical_portfolio_payload(window)
    version = active_version()
    if not version:
        raise ValueError("No active production model is available.")
    return _portfolio_payload_cached(version, active_model_signature(version), project_dataset_signature())


def portfolio_rows(window: str | None = None) -> list[dict]:
    return portfolio_payload(window)["items"]


def invalidate_portfolio_cache() -> None:
    _portfolio_payload_cached.cache_clear()


def summary(window: str | None = None) -> dict:
    if window:
        payload = portfolio_payload(window)
        predictions = payload["items"]
        manifest = payload["manifest"]
        levels = {key: 0 for key in ("critical", "high", "medium", "low")}
        exposure = {key: 0.0 for key in levels}
        sectors = set()
        scatter = []
        for prediction in predictions:
            level = prediction["risk_level"].lower(); levels[level] += 1
            exposure[level] += max(0.0, float(prediction["predicted_cost_overrun_amount_cr"] or 0)); sectors.add(prediction["sector"])
            observed = prediction["observed"]
            if observed["physical_progress_pct"] is not None and observed["financial_progress_pct"] is not None:
                group = "At Risk" if level in {"high", "critical"} else "Monitor" if level == "medium" else "On Track"
                scatter.append({"project_code": prediction["project_code"], "physical_progress_pct": observed["physical_progress_pct"], "financial_progress_pct": observed["financial_progress_pct"], "group": group})
        return {
            "projects": len(predictions), "original_cost_cr": round(sum(float(row["original_cost_cr"] or 0) for row in predictions), 2),
            "current_cost_basis_cr": round(sum(float(row["revised_cost_cr"] if row["revised_cost_cr"] is not None else row["original_cost_cr"] or 0) for row in predictions), 2),
            "expenditure_cr": round(sum(float(row["expenditure_cr"] or 0) for row in predictions), 2), "predicted_cost_exposure_cr": round(sum(exposure.values()), 2),
            "risk_distribution": levels, "cost_exposure_by_risk_cr": {key: round(value, 2) for key, value in exposure.items()}, "sectors": len(sectors),
            "dataset_snapshot": max((row["snapshot_date"] for row in predictions), default=None), "dataset_scope": f"Official PAIMANA frozen production holdout ledger for {window}; latest validated snapshot per project.",
            "model_version": manifest["model_version"], "model_scope": predictions[0]["model_scope"] if predictions else None, "inference_timestamp": payload["generated_at"], "expenditure_progress": scatter,
            "warning_drivers": [{"name": "Predicted delay", "count": sum(row["predicted_delay_days"] > 0 for row in predictions)}, {"name": "Physical progress unavailable", "count": sum(row["physical_progress_pct"] is None for row in predictions)}],
            "risk_trend": None, "risk_trend_status": "Unavailable: historical window view has no comparable monthly risk series.",
        }
    frame = projects_df()
    payload = portfolio_payload()
    predictions = payload["items"]
    levels = {key: 0 for key in ("critical", "high", "medium", "low")}
    exposure = {key: 0.0 for key in levels}
    for prediction in predictions:
        level = prediction["risk_level"].lower()
        levels[level] += 1
        exposure[level] += max(0.0, prediction["predicted_cost_overrun_amount_cr"])
    scatter = []
    for prediction in predictions:
        physical = prediction["observed"]["physical_progress_pct"]
        financial = prediction["observed"]["financial_progress_pct"]
        if physical is None or financial is None:
            continue
        group = "At Risk" if prediction["risk_level"] in {"HIGH", "CRITICAL"} else "Monitor" if prediction["risk_level"] == "MEDIUM" else "On Track"
        scatter.append({"project_code": prediction["project_code"], "physical_progress_pct": physical, "financial_progress_pct": financial, "group": group})
    warning_drivers = [
        {"name": "Predicted delay", "count": int(sum(prediction["predicted_delay_days"] > 0 for prediction in predictions))},
        {"name": "Progress not reported", "count": int(frame["physical_progress_pct"].isna().sum())},
        {"name": "Revised cost not reported", "count": int(frame["revised_cost_cr"].isna().sum())},
        {"name": "Revised date not reported", "count": int(frame["revised_end_date"].isna().sum())},
    ]
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce").max()
    return {
        "projects": int(len(frame)),
        "original_cost_cr": round(float(frame["original_cost_cr"].sum(min_count=1)), 2),
        "current_cost_basis_cr": round(float(frame["revised_cost_cr"].fillna(frame["original_cost_cr"]).sum(min_count=1)), 2),
        "expenditure_cr": round(float(frame["expenditure_cr"].sum(min_count=1)), 2),
        "predicted_cost_exposure_cr": round(sum(exposure.values()), 2),
        "risk_distribution": levels,
        "cost_exposure_by_risk_cr": {key: round(value, 2) for key, value in exposure.items()},
        "sectors": int(frame["sector"].nunique()),
        "dataset_snapshot": None if pd.isna(snapshot) else snapshot.strftime("%Y-%m-%d"),
        "dataset_scope": "curated official PAIMANA public-project subset",
        "model_version": active_version(),
        "model_scope": predictions[0]["model_scope"] if predictions else None,
        "inference_timestamp": payload["generated_at"],
        "expenditure_progress": scatter,
        "warning_drivers": warning_drivers,
        "risk_trend": None,
        "risk_trend_status": "Unavailable: the backend has no comparable historical portfolio-risk series.",
    }


SORT_FIELDS = {
    "name": "project_name", "project_name": "project_name",
    "code": "project_code", "project_code": "project_code",
    "sector": "sector", "cost": "predicted_cost_overrun_percentage",
    "predicted_cost_overrun_percentage": "predicted_cost_overrun_percentage",
    "time": "predicted_delay_days", "predicted_delay_days": "predicted_delay_days",
    "score": "risk_score", "risk_score": "risk_score",
}


def project_page(*, page: int, page_size: int, search: str | None, sector: str | None, ministry: str | None, risk_level: str | None, sort: str, direction: str, window: str | None = None) -> dict:
    all_rows = portfolio_rows(window)
    rows = all_rows
    if search:
        term = search.strip().casefold()
        rows = [row for row in rows if term in row["project_name"].casefold() or term in row["project_code"].casefold() or term in row["sector"].casefold() or term in (row["ministry"] or "").casefold()]
    if sector:
        rows = [row for row in rows if row["sector"] == sector]
    if ministry:
        rows = [row for row in rows if row["ministry"] == ministry]
    if risk_level:
        rows = [row for row in rows if row["risk_level"].casefold() == risk_level.casefold()]
    field = SORT_FIELDS[sort]
    rows = sorted(rows, key=lambda row: (row[field] is None, row[field] if row[field] is not None else 0), reverse=direction == "desc")
    total = len(rows)
    pages = max(1, math.ceil(total / page_size))
    selected_page = min(page, pages)
    start = (selected_page - 1) * page_size
    detail_only = {
        "cost_factors", "delay_factors", "risk_factors", "cost_explanation_status",
        "delay_explanation_status", "risk_explanation_status", "operational_drivers",
        "explanation_provenance", "cost_explanation_summary", "delay_explanation_summary",
        "risk_explanation_summary",
    }
    page_rows = [{key: value for key, value in row.items() if key not in detail_only} for row in rows[start:start + page_size]]
    levels = {key: 0 for key in ("critical", "high", "medium", "low")}
    exposure = {key: 0.0 for key in levels}
    for row in all_rows:
        level = row["risk_level"].lower()
        levels[level] += 1
        exposure[level] += max(0.0, row["predicted_cost_overrun_amount_cr"])
    return {
        "items": page_rows, "total": total, "page": selected_page,
        "page_size": page_size, "pages": pages,
        "sectors": sorted({row["sector"] for row in all_rows}),
        "ministries": sorted({row["ministry"] for row in all_rows if row["ministry"]}),
        "risk_distribution": levels,
        "cost_exposure_by_risk_cr": {key: round(value, 2) for key, value in exposure.items()},
        "predicted_cost_exposure_cr": round(sum(exposure.values()), 2),
        "model_version": all_rows[0]["model_version"] if all_rows else active_version(),
        "dataset_snapshot": all_rows[0]["snapshot_date"] if all_rows else None,
        "inference_timestamp": all_rows[0]["inference_timestamp"] if all_rows else portfolio_payload()["generated_at"],
    }
