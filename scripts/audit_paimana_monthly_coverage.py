#!/usr/bin/env python3
"""Audit live PAIMANA archive coverage without modifying production datasets.

This script is intentionally read-only. It discovers every public flash report,
probes one representative report per financial year with the current monthly
parser, and reports how much of the checked-in monthly history can be linked to
completed outcomes by project id or conservative normalized project name.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.paimana_ingestion_service import (
    _fetch,
    _report_month,
    discover_archive_reports,
    extract_report_text,
    parse_project_list,
)

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

    print("ARCHIVE_REPORT_COUNT=" + str(len(reports)))
    print("ARCHIVE_FINANCIAL_YEARS=" + json.dumps(sorted(by_year)))
    print("ARCHIVE_REPORTS_PER_YEAR=" + json.dumps({k: len(v) for k, v in sorted(by_year.items())}, sort_keys=True))

    probe_counts: dict[str, dict] = {}
    for financial_year, candidates in sorted(by_year.items()):
        dated = [(r, _report_month(r["financial_year"], r["label"])) for r in candidates]
        dated = [(r, month) for r, month in dated if month is not None]
        if not dated:
            continue
        # Prefer the latest dated report in the financial year; this maximizes
        # the chance that the annual report contains the full ongoing-project table.
        report, report_month = max(dated, key=lambda pair: pair[1])
        try:
            payload = _fetch(report["url"])
            if not payload.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
                handle.write(payload)
                handle.flush()
                text = extract_report_text(Path(handle.name))
            parsed = parse_project_list(text, report_month, report["label"], report["url"])
            probe_counts[financial_year] = {
                "label": report["label"],
                "month": report_month.strftime("%Y-%m-%d"),
                "rows_with_current_parser": int(len(parsed)),
            }
        except Exception as exc:
            probe_counts[financial_year] = {
                "label": report["label"],
                "month": report_month.strftime("%Y-%m-%d"),
                "error": f"{type(exc).__name__}: {exc}",
            }
    print("ARCHIVE_PARSER_PROBE=" + json.dumps(probe_counts, sort_keys=True))

    monthly = pd.read_csv(MONTHLY, dtype={"project_id": str})
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": str})
    monthly_ids = set(monthly.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"}
    outcome_ids = set(outcomes.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) - {"", "nan"}
    id_overlap = monthly_ids & outcome_ids

    monthly_names = Counter(_key(v) for v in monthly.get("project_name", pd.Series(dtype=str)))
    outcome_names = Counter(_key(v) for v in outcomes.get("project_name", pd.Series(dtype=str)))
    unique_name_overlap = {
        name for name in monthly_names.keys() & outcome_names.keys()
        if name and monthly_names[name] >= 1 and outcome_names[name] == 1
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
    print("CHECKED_IN_JOIN_COVERAGE=" + json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
