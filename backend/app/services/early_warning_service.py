"""Early-warning view derived from the frozen monthly lifecycle ledger.

This consumes the same PAIMANA ledger and pre-holdout severity calibration as
the historical portfolio API. Scores are never generated in the browser.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.app.services.range_portfolio_service import (
    _manifest,
    _paths,
)

LEVEL_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _number(value: object, digits: int = 2) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def _status(previous: dict | None, current: dict) -> str:
    if previous is None:
        return "New Escalation" if LEVEL_RANK[current["risk_level"]] >= 3 else "Persistent"
    old_level, new_level = LEVEL_RANK[previous["risk_level"]], LEVEL_RANK[current["risk_level"]]
    if new_level > old_level:
        return "New Escalation"
    if current["risk_score"] > previous["risk_score"]:
        return "Worsening"
    if current["risk_score"] < previous["risk_score"]:
        return "Improving"
    return "Persistent"


@lru_cache(maxsize=3)
def _ledger(window: str) -> pd.DataFrame:
    _, ledger_path = _paths(window)
    frame = pd.read_csv(ledger_path, dtype={"canonical_project_id": str}, low_memory=False)
    frame["canonical_project_id"] = frame["canonical_project_id"].astype(str)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_date"]).sort_values(["canonical_project_id", "snapshot_date"])
    frame["risk_level"] = frame["predicted_risk"].fillna("LOW").astype(str).str.upper()
    frame.loc[~frame["risk_level"].isin(LEVEL_RANK), "risk_level"] = "LOW"
    # The frozen model emits a category, not a probability. The UI's 0-100
    # score is its documented ordinal representation for display/sorting only.
    frame["risk_score"] = frame["risk_level"].map(lambda level: float(LEVEL_RANK[level] * 25))
    return frame


def _contexts(window: str, keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """Use the persisted, hash-linked portfolio view for display context."""
    path = Path(__file__).resolve().parents[3] / "data" / "processed" / "portfolio_windows" / f"{window}.json"
    document = json.loads(path.read_text())
    return {
        (str(item["project_code"]), str(item["snapshot_date"])): item
        for item in document["payload"]["items"]
        if (str(item["project_code"]), str(item["snapshot_date"])) in keys
    }


def _row(record: pd.Series, context: dict, previous: dict | None, first_appeared: str | None) -> dict:
    approved, cost = _number(context.get("approved_cost_cr")), _number(record.get("predicted_cost_overrun"))
    delay_days, score = _number(record.get("predicted_delay_days"), 1), float(record["risk_score"])
    old_score = None if previous is None else previous["risk_score"]
    change = None if old_score is None else round(score - old_score, 1)
    return {
        "project_id": str(record["canonical_project_id"]), "project_name": str(record["project_name"]),
        "sector": str(context.get("sector") or "Not reported"),
        "ministry": None if pd.isna(context.get("ministry")) else context.get("ministry"),
        "snapshot_date": record["snapshot_date"].strftime("%Y-%m-%d"),
        "previous_risk": None if previous is None else {"level": previous["risk_level"].title(), "score": old_score},
        "current_risk": {"level": record["risk_level"].title(), "score": score}, "risk_change": change,
        "predicted_cost_overrun_percentage": cost,
        "predicted_delay_months": None if delay_days is None else round(delay_days / 30.4375, 1),
        "predicted_delay_days": delay_days,
        "predicted_cost_overrun_amount_cr": None if cost is None or approved is None else round(approved * cost / 100, 2),
        "warning_status": _status(previous, {"risk_level": record["risk_level"], "risk_score": score}),
        "threshold_crossing": previous is not None and LEVEL_RANK[record["risk_level"]] > LEVEL_RANK[previous["risk_level"]],
        "first_appeared": first_appeared, "priority": record["risk_level"].title(),
        "trigger": f"Predicted delay: {delay_days:.0f} days" if delay_days is not None else "Predicted delay unavailable",
    }


def _selected_records(frame: pd.DataFrame, as_of: pd.Timestamp) -> list[tuple[pd.Series, pd.Series | None, str | None]]:
    selected = []
    for _, project in frame.groupby("canonical_project_id", sort=False):
        eligible = project[project["snapshot_date"] <= as_of]
        if eligible.empty:
            continue
        current, previous = eligible.iloc[-1], eligible.iloc[-2] if len(eligible) > 1 else None
        elevated = eligible[eligible["risk_level"].isin(["HIGH", "CRITICAL"])]
        first = None if elevated.empty else elevated.iloc[0]["snapshot_date"].strftime("%Y-%m-%d")
        selected.append((current, previous, first))
    return selected


def _monthly_counts(frame: pd.DataFrame, dates: list[pd.Timestamp]) -> list[dict]:
    points = []
    for date in dates:
        records = _selected_records(frame, date)
        statuses = [_status(None if previous is None else {"risk_level": previous["risk_level"], "risk_score": float(previous["risk_score"])}, {"risk_level": current["risk_level"], "risk_score": float(current["risk_score"])}) for current, previous, _ in records]
        changes = [float(current["risk_score"] - previous["risk_score"]) for current, previous, _ in records if previous is not None]
        points.append({"month": date.strftime("%b %Y"), "snapshot_date": date.strftime("%Y-%m-%d"), "newEsc": statuses.count("New Escalation"), "worsening": statuses.count("Worsening"), "persistent": statuses.count("Persistent"), "improving": statuses.count("Improving"), "average_risk_change": round(sum(changes) / len(changes), 1) if changes else 0.0})
    return points


def _drivers(window: str) -> tuple[list[dict], dict]:
    manifest_path, _ = _paths(window)
    path = manifest_path.parent / "shap_importance.json"
    if not path.exists():
        return [], {"available": False, "reason": "The production bundle has no persisted global feature-importance artifact."}
    risk = json.loads(path.read_text()).get("risk", {})
    features = risk.get("features", [])[:6]
    return [{"name": str(feature["feature"]), "value": round(float(feature["importance"]) * 100, 2)} for feature in features], {"available": bool(features), "method": risk.get("method"), "scope": risk.get("scope"), "source": "models/monthly_lifecycle/2001_2022/shap_importance.json"}


def build_early_warnings(window: str = "2001_2022", *, project_id: str | None = None, as_of: str | None = None, search: str | None = None, severity: str | None = None, sector: str | None = None, ministry: str | None = None, threshold_only: bool = False) -> dict:
    frame, dates = _ledger(window), None
    dates = sorted(frame["snapshot_date"].drop_duplicates())
    if not dates:
        raise ValueError("The frozen evaluation ledger has no valid snapshot dates.")
    requested = pd.to_datetime(as_of, errors="coerce") if as_of else dates[-1]
    if pd.isna(requested):
        raise ValueError("The requested snapshot date is invalid.")
    valid = [date for date in dates if date <= requested]
    if not valid:
        raise ValueError("No frozen evaluation snapshot is available on or before the requested date.")
    selected_date = valid[-1]
    selected = _selected_records(frame, selected_date)
    contexts = _contexts(window, {(str(current["canonical_project_id"]), current["snapshot_date"].strftime("%Y-%m-%d")) for current, _, _ in selected})
    rows = [_row(current, contexts.get((str(current["canonical_project_id"]), current["snapshot_date"].strftime("%Y-%m-%d")), {}), None if previous is None else {"risk_level": previous["risk_level"], "risk_score": float(previous["risk_score"])}, first) for current, previous, first in selected]
    all_rows = list(rows)
    if project_id:
        rows = [row for row in rows if row["project_id"] == project_id]
    if search:
        needle = search.casefold(); rows = [row for row in rows if needle in row["project_name"].casefold() or needle in row["sector"].casefold() or needle in str(row["ministry"] or "").casefold() or needle in row["trigger"].casefold()]
    if severity and severity != "All": rows = [row for row in rows if row["warning_status"] == severity]
    if sector and sector != "All Sectors": rows = [row for row in rows if row["sector"] == sector]
    if ministry and ministry != "All Ministries": rows = [row for row in rows if row["ministry"] == ministry]
    if threshold_only: rows = [row for row in rows if row["threshold_crossing"]]
    rows.sort(key=lambda row: (LEVEL_RANK[row["current_risk"]["level"].upper()], row["current_risk"]["score"], row["risk_change"] or 0), reverse=True)
    trend_dates = [date for date in dates if date <= selected_date][-6:]
    trend = _monthly_counts(frame, trend_dates)
    previous_trend = trend[-2] if len(trend) > 1 else {"newEsc": 0, "persistent": 0}
    critical = sum(row["current_risk"]["level"] == "Critical" for row in all_rows)
    persistent = sum(row["warning_status"] == "Persistent" and row["current_risk"]["level"] in {"High", "Critical"} for row in all_rows)
    improved = sum(row["warning_status"] == "Improving" for row in all_rows)
    changes = [row["risk_change"] for row in all_rows if row["risk_change"] is not None]
    kpis = [
        {"key": "new", "value": trend[-1]["newEsc"], "delta": f"vs previous snapshot: {trend[-1]['newEsc'] - previous_trend['newEsc']:+d}", "data": [point["newEsc"] for point in trend]},
        {"key": "critical", "value": critical, "delta": "current model risk category", "data": [sum(1 for current, _, _ in _selected_records(frame, date) if current["risk_level"] == "CRITICAL") for date in trend_dates]},
        {"key": "persistent", "value": persistent, "delta": f"vs previous snapshot: {trend[-1]['persistent'] - previous_trend['persistent']:+d}", "data": [point["persistent"] for point in trend]},
        {"key": "improved", "value": improved, "delta": "risk score decreased since prior snapshot", "data": [point["improving"] for point in trend]},
        {"key": "average_change", "value": round(sum(changes) / len(changes), 1) if changes else None, "delta": "points since prior snapshot", "data": [point["average_risk_change"] for point in trend]},
    ]
    timeline = None
    if rows:
        focus = rows[0]; history = frame[(frame["canonical_project_id"] == focus["project_id"]) & (frame["snapshot_date"] <= selected_date)].tail(6)
        timeline = {"project_id": focus["project_id"], "project_name": focus["project_name"], "points": [{"month": record.snapshot_date.strftime("%b %Y"), "score": float(record.risk_score)} for record in history.itertuples()], "risk_change": focus["risk_change"], "current_risk": focus["current_risk"]}
    alerts = [{"project_name": row["project_name"], "date": row["snapshot_date"], "text": (f"Risk moved from {row['previous_risk']['level']} to {row['current_risk']['level']}; " if row["previous_risk"] and row["previous_risk"]["level"] != row["current_risk"]["level"] else f"Risk score changed by {row['risk_change']:+.1f} points; " if row["risk_change"] is not None else "First evaluated snapshot; ") + row["trigger"]} for row in rows[:4]]
    drivers, driver_metadata, manifest = _drivers(window)[0], _drivers(window)[1], _manifest(window)
    return {"snapshot_date": selected_date.strftime("%Y-%m-%d"), "available_snapshots": [date.strftime("%Y-%m-%d") for date in dates[-12:]], "summary": {"kpis": kpis, "total_projects": len(all_rows), "matching_projects": len(rows)}, "warnings": rows, "trend": trend, "drivers": drivers, "drivers_metadata": driver_metadata, "timeline": timeline, "recent_alerts": alerts, "available_filters": {"sectors": sorted({row["sector"] for row in all_rows}), "ministries": sorted({row["ministry"] for row in all_rows if row["ministry"]})}, "source": {"window": window, "model_version": manifest["model_version"], "run_id": manifest["run_id"], "ledger": "models/monthly_lifecycle/2001_2022/prediction_validation.csv", "trajectories": "data/processed/paimana_project_trajectories.csv"}}
