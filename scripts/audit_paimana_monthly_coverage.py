#!/usr/bin/env python3
"""Audit public PAIMANA archive and checked-in monthly/outcome join coverage.

The audit is read-only and intentionally does not download every historical PDF;
that would make CI depend on dozens of large remote files. Historical PDF parsing
is exercised separately by ingestion tests and the live ingestion experiment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.paimana_ingestion_service import discover_archive_reports

MONTHLY = ROOT / "data" / "processed" / "project_monthly_history.csv"
OUTCOMES = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"


def _key(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def main() -> None:
    reports = discover_archive_reports()
    by_year: dict[str, list[dict]] = defaultdict(list)
    for report in reports:
        by_year[report["financial_year"]].append(report)
    labels_by_year = {year: sorted({item["label"] for item in rows}) for year, rows in sorted(by_year.items())}
    print("ARCHIVE_REPORT_COUNT=" + str(len(reports)), flush=True)
    print("ARCHIVE_FINANCIAL_YEARS=" + json.dumps(sorted(by_year)), flush=True)
    print("ARCHIVE_REPORTS_PER_YEAR=" + json.dumps({k: len(v) for k, v in sorted(by_year.items())}, sort_keys=True), flush=True)
    print("ARCHIVE_LABELS_PER_YEAR=" + json.dumps(labels_by_year, sort_keys=True), flush=True)

    monthly = pd.read_csv(MONTHLY, dtype={"project_id": str})
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": str})
    monthly_ids = set(monthly.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"}
    outcome_ids = set(outcomes.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"}
    id_overlap = monthly_ids & outcome_ids

    monthly_names = Counter(_key(v) for v in monthly.get("project_name", pd.Series(dtype=str)))
    outcome_names = Counter(_key(v) for v in outcomes.get("project_name", pd.Series(dtype=str)))
    unique_name_overlap = {
        name for name in monthly_names.keys() & outcome_names.keys()
        if name and outcome_names[name] == 1
    }
    matched_outcome_names = outcomes.get("project_name", pd.Series(dtype=str)).map(_key).isin(unique_name_overlap)
    matched_outcome_ids = outcomes.get("project_id", pd.Series(index=outcomes.index, dtype=str)).fillna("").astype(str).isin(id_overlap)

    coverage = {
        "monthly_rows": int(len(monthly)),
        "monthly_projects": int(monthly.project_id.nunique()),
        "monthly_min_month": str(monthly.month.min()),
        "monthly_max_month": str(monthly.month.max()),
        "completed_rows": int(len(outcomes)),
        "completed_rows_with_project_id": int(outcomes.get("project_id", pd.Series(dtype=str)).notna().sum()),
        "exact_project_id_overlap": int(len(id_overlap)),
        "completed_rows_matched_by_id_or_unique_normalized_name": int((matched_outcome_ids | matched_outcome_names).sum()),
        "unique_normalized_name_overlap": int(len(unique_name_overlap)),
    }
    print("CHECKED_IN_JOIN_COVERAGE=" + json.dumps(coverage, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
