#!/usr/bin/env python3
"""Evaluate whether the July 2026 PAIMANA snapshot supports a true MAE test.

This script is intentionally read-only with respect to production artifacts. It
fetches only the published July 2026 report into temporary runner storage,
parses both the project snapshot and projects completed during July, resolves
identities with the production lifecycle code, and applies the exact
leakage-safe rule: snapshot_date < completion_date.

If no pre-completion July rows have realized cost and delay outcomes, true MAE
is scientifically unavailable and the script reports null rather than using a
same-snapshot proxy target.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from backend.app.ml.monthly_lifecycle import TARGETS, resolve_identities
from backend.app.services.paimana_ingestion_service import (
    _fetch,
    extract_report_text,
    parse_project_list,
)
from scripts.ingest_paimana_completed_reports import parse_completed_projects

BASE = "https://paimana-proj.mospi.gov.in"
ARCHIVE_PATH = "/ArchiveReport/flash/2026-27/FlashReport_July_2026.pdf"
# The first URL is the direct published archive path. The two ViewPdf forms
# mirror PAIMANA's archive-link convention and avoid touching the archive index.
CANDIDATE_URLS = [
    f"{BASE}{ARCHIVE_PATH}",
    f"{BASE}/ReportPage/ViewPdf?path={quote(ARCHIVE_PATH)}",
    f"{BASE}/ReportPage/ViewPdf?id=&path={quote(ARCHIVE_PATH)}",
]


def _download_july_pdf() -> tuple[str, bytes]:
    errors: list[str] = []
    for url in CANDIDATE_URLS:
        try:
            payload = _fetch(url)
            if payload.startswith(b"%PDF"):
                return url, payload
            errors.append(f"{url}: response was not a PDF ({len(payload)} bytes)")
        except Exception as exc:  # surface every attempted official route
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not download official July 2026 PAIMANA PDF. " + " | ".join(errors))


def main() -> None:
    source_url, pdf_bytes = _download_july_pdf()
    report_month = pd.Timestamp(year=2026, month=7, day=31)

    with tempfile.TemporaryDirectory(prefix="paimana-july-2026-") as tmp:
        pdf = Path(tmp) / "FlashReport_July_2026.pdf"
        pdf.write_bytes(pdf_bytes)
        text = extract_report_text(pdf)
        snapshots = parse_project_list(
            text,
            report_month=report_month,
            source_report="July 2026",
            source_url=source_url,
        )
        outcomes = parse_completed_projects(text, source_url, "2026-27")

    if snapshots.empty:
        raise SystemExit("July 2026 report downloaded, but no project snapshot rows were parsed.")

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
        "source_url": source_url,
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


if __name__ == "__main__":
    main()
