#!/usr/bin/env python3
"""Audit PAIMANA monthly-history coverage for Experiment 2.

The checked-in official monthly data is audited first and always produces a
machine-readable report. Live archive availability is supplementary: an outage
must never hide the local coverage numbers or make a reproducible experiment
silently depend on the network.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.paimana_ingestion_service import archive_discovery_status

MONTHLY = ROOT / "data" / "processed" / "project_monthly_history.csv"
OUTCOMES = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
REPORT = ROOT / "reports" / "experiments" / "exp2_monthly_history_coverage.json"


def _key(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _numeric(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def checked_in_coverage(monthly: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    monthly_ids = set(monthly.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"}
    outcome_ids = set(outcomes.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"}
    id_overlap = monthly_ids & outcome_ids

    monthly_names = Counter(_key(v) for v in monthly.get("project_name", pd.Series(dtype=str)))
    outcome_names = Counter(_key(v) for v in outcomes.get("project_name", pd.Series(dtype=str)))
    unique_name_overlap = {
        name for name in monthly_names.keys() & outcome_names.keys()
        if name and outcome_names[name] == 1
    }

    monthly_cost = _numeric(monthly, "original_cost", "approved_cost_cr")
    outcome_cost = _numeric(outcomes, "approved_cost_cr", "original_cost", "original_cost_cr")
    monthly_lookup: dict[str, float] = {}
    for name, values in pd.DataFrame({"name": monthly.project_name.map(_key), "cost": monthly_cost}).groupby("name"):
        numeric = values.cost.dropna()
        if name and len(numeric):
            monthly_lookup[name] = float(numeric.median())

    strict_name_matches = 0
    for index, row in outcomes.iterrows():
        name = _key(row.get("project_name"))
        if name not in unique_name_overlap:
            continue
        left = monthly_lookup.get(name)
        right = outcome_cost.loc[index]
        if left is None or pd.isna(right) or right <= 0:
            continue
        relative_difference = abs(left - float(right)) / float(right)
        if relative_difference <= 0.05:
            strict_name_matches += 1

    matched_by_id = outcomes.get("project_id", pd.Series(index=outcomes.index, dtype=str)).fillna("").astype(str).isin(id_overlap)
    matched_by_name = outcomes.get("project_name", pd.Series(dtype=str)).map(_key).isin(unique_name_overlap)

    months = pd.to_datetime(monthly.get("month"), errors="coerce")
    per_project = monthly.groupby("project_id", dropna=True).size() if "project_id" in monthly else pd.Series(dtype=int)
    lifecycle_non_null = {}
    for column in [
        "revised_cost", "current_expenditure", "physical_progress_percentage",
        "financial_progress_percentage", "planned_start_date", "planned_completion_date",
        "revised_completion_date", "implementing_agency", "sector", "ministry", "state",
    ]:
        if column in monthly:
            lifecycle_non_null[column] = {
                "non_null_rows": int(monthly[column].notna().sum()),
                "coverage_percent": round(float(monthly[column].notna().mean() * 100), 2),
            }

    return {
        "monthly_rows": int(len(monthly)),
        "monthly_projects": int(monthly.project_id.nunique()) if "project_id" in monthly else 0,
        "monthly_min_month": None if months.isna().all() else months.min().strftime("%Y-%m-%d"),
        "monthly_max_month": None if months.isna().all() else months.max().strftime("%Y-%m-%d"),
        "multi_snapshot_projects": int((per_project > 1).sum()),
        "max_snapshots_per_project": int(per_project.max()) if len(per_project) else 0,
        "completed_rows": int(len(outcomes)),
        "completed_rows_with_project_id": int(outcomes.get("project_id", pd.Series(dtype=str)).notna().sum()),
        "exact_project_id_overlap": int(len(id_overlap)),
        "completed_rows_matched_by_id": int(matched_by_id.sum()),
        "unique_normalized_name_overlap": int(len(unique_name_overlap)),
        "completed_rows_matched_by_id_or_unique_normalized_name": int((matched_by_id | matched_by_name).sum()),
        "strict_name_and_cost_matches_within_5_percent": int(strict_name_matches),
        "lifecycle_field_coverage": lifecycle_non_null,
    }


def main() -> None:
    monthly = pd.read_csv(MONTHLY, dtype={"project_id": str})
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": str})
    coverage = checked_in_coverage(monthly, outcomes)
    print("CHECKED_IN_JOIN_COVERAGE=" + json.dumps(coverage, sort_keys=True), flush=True)

    discovery = archive_discovery_status(timeout=8)
    reports = discovery["reports"]
    by_year: dict[str, list[dict]] = defaultdict(list)
    for report in reports:
        by_year[str(report.get("financial_year") or "unknown")].append(report)
    source_summary = {
        "source": discovery["source"],
        "error": discovery["error"],
        "report_count": len(reports),
        "financial_years": sorted(by_year),
        "reports_per_year": {key: len(value) for key, value in sorted(by_year.items())},
    }
    print("ARCHIVE_DISCOVERY_STATUS=" + json.dumps(source_summary, sort_keys=True), flush=True)

    report = {
        "experiment": "hybrid_cost_regime_lifecycle_coverage",
        "checked_in": coverage,
        "archive_discovery": source_summary,
        "training_join_policy": {
            "preferred": "exact official project_id",
            "fallback": "unique normalized project name plus independent cost/date corroboration",
            "ambiguous_names_allowed": False,
            "synthetic_ids_allowed": False,
            "post_completion_snapshots_allowed": False,
            "milestone_delay_used": False,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print("COVERAGE_REPORT=" + str(REPORT.relative_to(ROOT)), flush=True)


if __name__ == "__main__":
    main()
