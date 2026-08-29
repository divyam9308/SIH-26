#!/usr/bin/env python3
"""Evaluate whether the July 2026 PAIMANA snapshot supports a true MAE test.

This script is intentionally read-only with respect to production artifacts. It
fetches the official July 2026 report into a temporary directory, parses both
the project snapshot and the projects completed during July, resolves identities
with the same lifecycle code used by production, and applies the exact
leakage-safe rule: snapshot_date < completion_date.

If no pre-completion July rows have realized cost and delay outcomes, true MAE
is scientifically unavailable and the script reports null rather than using a
same-snapshot proxy target.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from backend.app.ml.monthly_lifecycle import TARGETS, resolve_identities
from backend.app.services.paimana_ingestion_service import (
    _fetch,
    discover_archive_reports,
    extract_report_text,
    parse_project_list,
)
from scripts.ingest_paimana_completed_reports import parse_completed_projects


def main() -> None:
    reports = [
        report
        for report in discover_archive_reports()
        if report.get("calendar_year") == 2026
        and report.get("report_month") == "July"
    ]
    if not reports:
        raise SystemExit("Official July 2026 PAIMANA Flash Report was not discovered.")

    snapshot_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []
    source_urls: list[str] = []

    with tempfile.TemporaryDirectory(prefix="paimana-july-2026-") as tmp:
        tmpdir = Path(tmp)
        for index, report in enumerate(reports):
            source_url = report.get("source_url", report.get("url", ""))
            source_urls.append(source_url)
            pdf = tmpdir / f"july_2026_{index}.pdf"
            pdf.write_bytes(_fetch(source_url))
            text = extract_report_text(pdf)
            report_month = pd.Timestamp(year=2026, month=7, day=31)
            snapshots = parse_project_list(
                text,
                report_month=report_month,
                source_report=report.get("report_label", "July 2026"),
                source_url=source_url,
            )
            completed = parse_completed_projects(text, source_url, report["financial_year"])
            if not snapshots.empty:
                snapshot_frames.append(snapshots)
            if not completed.empty:
                outcome_frames.append(completed)

    snapshots = (
        pd.concat(snapshot_frames, ignore_index=True)
        if snapshot_frames
        else pd.DataFrame()
    )
    outcomes = (
        pd.concat(outcome_frames, ignore_index=True)
        if outcome_frames
        else pd.DataFrame()
    )

    if snapshots.empty:
        raise SystemExit("July 2026 report was found but no project snapshot rows were parsed.")

    # Deduplicate multipart report output before identity resolution.
    if "project_id" in snapshots.columns:
        snapshots = snapshots.drop_duplicates(["project_id", "snapshot_date"], keep="last")
    if not outcomes.empty:
        outcomes = outcomes.copy()
        outcomes["project_id"] = outcomes["project_id"].replace("", pd.NA)
        coded = outcomes[outcomes.project_id.notna()].drop_duplicates("project_id", keep="last")
        legacy = outcomes[outcomes.project_id.isna()].copy()
        if not legacy.empty:
            legacy["_name"] = legacy.project_name.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
            legacy = legacy.drop_duplicates(["_name", "completion_date"], keep="last").drop(columns="_name")
        outcomes = pd.concat([coded, legacy], ignore_index=True)

    if outcomes.empty:
        eligible = pd.DataFrame()
        resolved = snapshots.copy()
        verified = 0
    else:
        resolved, _identity = resolve_identities(snapshots, outcomes)
        snapshot_date = pd.to_datetime(resolved["snapshot_date"], errors="coerce")
        completion_date = pd.to_datetime(resolved["completion_date"], errors="coerce")
        target_ok = resolved[TARGETS[:2]].notna().all(axis=1)
        identity_ok = resolved["identity_verified"].fillna(False).astype(bool)
        eligible = resolved[identity_ok & target_ok & snapshot_date.lt(completion_date)].copy()
        verified = int(identity_ok.sum())

    payload = {
        "period": "2026-07",
        "official_reports_found": len(reports),
        "source_urls": source_urls,
        "july_snapshot_rows": int(len(snapshots)),
        "july_completed_outcomes_parsed": int(len(outcomes)),
        "identity_verified_july_rows": verified,
        "leakage_safe_labeled_rows": int(len(eligible)),
        "evaluation_rule": "identity_verified and cost/delay targets present and snapshot_date < completion_date",
        "cost_mae": None,
        "delay_mae_days": None,
        "status": "MAE_UNAVAILABLE_NO_LEAKAGE_SAFE_REALIZED_LABELS" if eligible.empty else "LABELS_AVAILABLE_MODEL_SCORING_REQUIRED",
    }
    print("JULY_2026_EVALUATION=" + json.dumps(payload, sort_keys=True))

    # A non-zero eligible cohort means a true model scoring stage can be added;
    # zero is an expected scientific outcome and must remain a green check.


if __name__ == "__main__":
    main()
