"""Evidence-only operational drivers for a single PAIMANA project snapshot.

These rules deliberately describe observed project conditions, not model feature
importance and not unrecorded causal explanations.  The processed PAIMANA
schemas currently contain no implementation-constraint or remarks columns, so
no named causes (for example land acquisition or contractor delay) are inferred
here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


MAX_DRIVERS = 5
MATERIAL_EXTENSION_DAYS = 90
MATERIAL_COST_REVISION_PCT = 5.0
MATERIAL_PROGRESS_GAP_POINTS = 20.0
MATERIAL_PROGRESS_LAG_POINTS = 20.0
STAGNANT_PROGRESS_POINTS = 1.0
STAGNANT_PERIOD_DAYS = 90


def _value(record: Mapping[str, Any] | pd.Series, *names: str) -> Any:
    for name in names:
        value = record.get(name) if isinstance(record, Mapping) else record.get(name)
        if value is not None and not pd.isna(value):
            return value
    return None


def _number(record: Mapping[str, Any] | pd.Series, *names: str) -> float | None:
    value = pd.to_numeric(_value(record, *names), errors="coerce")
    return None if pd.isna(value) else float(value)


def _date(record: Mapping[str, Any] | pd.Series, *names: str) -> pd.Timestamp | None:
    value = pd.to_datetime(_value(record, *names), errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value)


def _driver(
    kind: str, label: str, category: str, evidence: str, *, source: str, rank: int,
) -> dict[str, str | int]:
    return {
        "type": kind,
        "label": label,
        "category": category,
        "evidence": evidence,
        "provenance": "derived",
        "source": source,
        "_rank": rank,
    }


def operational_drivers(
    record: Mapping[str, Any] | pd.Series,
    history: pd.DataFrame | None = None,
    *,
    source: str,
) -> list[dict[str, str]]:
    """Return up to five deterministic, evidence-backed operational signals.

    ``record`` must be the selected snapshot.  ``history``, if supplied, must
    already be restricted to that project and evaluation snapshot.  This keeps
    frozen views from accidentally consulting the current/live project data.
    """
    results: list[dict[str, str | int]] = []
    planned = _date(record, "planned_completion_date", "original_end_date")
    revised = _date(record, "revised_completion_date", "revised_end_date")
    snapshot = _date(record, "snapshot_date")
    actual = _date(record, "actual_completion_date", "completion_date")
    physical = _number(record, "physical_progress", "physical_progress_pct")
    financial = _number(record, "financial_progress", "financial_progress_pct")
    expected = _number(record, "expected_progress_percentage")
    approved = _number(record, "approved_cost_cr", "original_cost_cr")
    revised_cost = _number(record, "revised_cost_cr")

    if planned is not None and revised is not None:
        extension = int((revised - planned).days)
        if extension >= MATERIAL_EXTENSION_DAYS:
            results.append(_driver(
                "SCHEDULE_EXTENSION", "Schedule extension", "DELAY",
                f"Revised completion is {extension:,} days later than the original schedule.",
                source=source, rank=90,
            ))

    if approved is not None and approved > 0 and revised_cost is not None:
        revision_pct = (revised_cost - approved) / approved * 100
        if revision_pct >= MATERIAL_COST_REVISION_PCT:
            results.append(_driver(
                "COST_REVISION", "Cost revision", "COST",
                f"Revised cost is {revision_pct:.1f}% above the approved cost.",
                source=source, rank=80,
            ))

    if physical is not None and financial is not None:
        gap = financial - physical
        if gap >= MATERIAL_PROGRESS_GAP_POINTS:
            results.append(_driver(
                "EXPENDITURE_PROGRESS_MISMATCH", "Expenditure-progress mismatch", "IMPLEMENTATION",
                f"Financial progress exceeds physical progress by {gap:.1f} percentage points.",
                source=source, rank=85,
            ))

    if physical is not None and expected is not None:
        lag = expected - physical
        if lag >= MATERIAL_PROGRESS_LAG_POINTS:
            results.append(_driver(
                "PHYSICAL_PROGRESS_LAG", "Physical progress lag", "DELAY",
                f"Physical progress is {lag:.1f} percentage points below the recorded expected lifecycle progress.",
                source=source, rank=88,
            ))

    if planned is not None and snapshot is not None and snapshot > planned and actual is None and physical is not None and physical < 100:
        overdue = int((snapshot - planned).days)
        results.append(_driver(
            "PLANNED_DEADLINE_CROSSED", "Planned deadline crossed", "DELAY",
            f"The original completion deadline passed {overdue:,} days before this snapshot while reported physical progress was {physical:.1f}%.",
            source=source, rank=95,
        ))

    if history is not None and not history.empty:
        work = history.copy()
        work["snapshot_date"] = pd.to_datetime(work.get("snapshot_date"), errors="coerce")
        progress_column = "physical_progress" if "physical_progress" in work else "physical_progress_pct" if "physical_progress_pct" in work else None
        if progress_column:
            progress = pd.to_numeric(work[progress_column], errors="coerce")
            observed = work.assign(_progress=progress).dropna(subset=["snapshot_date", "_progress"]).sort_values("snapshot_date").drop_duplicates("snapshot_date", keep="last")
            if len(observed) >= 2:
                first, last = observed.iloc[0], observed.iloc[-1]
                days = int((last.snapshot_date - first.snapshot_date).days)
                gain = float(last._progress - first._progress)
                if days >= STAGNANT_PERIOD_DAYS and gain <= STAGNANT_PROGRESS_POINTS:
                    results.append(_driver(
                        "STAGNANT_PROGRESS", "Stagnant physical progress", "IMPLEMENTATION",
                        f"Physical progress changed by only {gain:.1f} percentage points across {days:,} days of official reporting.",
                        source=source, rank=75,
                    ))
        date_column = "revised_completion_date" if "revised_completion_date" in work else "revised_end_date" if "revised_end_date" in work else None
        if date_column:
            dates = pd.to_datetime(work[date_column], errors="coerce")
            values = work.assign(_revised_date=dates).dropna(subset=["snapshot_date", "_revised_date"]).sort_values("snapshot_date").drop_duplicates("snapshot_date", keep="last")["_revised_date"]
            changes = int(values.ne(values.shift()).sum() - (1 if len(values) else 0))
            if changes >= 2:
                results.append(_driver(
                    "REPEATED_COMPLETION_REVISION", "Repeated completion-date revisions", "DELAY",
                    f"The revised completion date changed {changes} times across official snapshots.",
                    source=source, rank=82,
                ))

    # One signal per type, strongest first, with a stable tie-breaker.
    unique = {item["type"]: item for item in results}
    ranked = sorted(unique.values(), key=lambda item: (-int(item["_rank"]), str(item["type"])))[:MAX_DRIVERS]
    return [{key: value for key, value in item.items() if key != "_rank"} for item in ranked]  # type: ignore[return-value]
