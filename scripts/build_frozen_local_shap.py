"""Warm the verified frozen local-SHAP cache without retraining models.

Only projects whose exact frozen source row reproduces Cost, Delay, and Risk
are written.  It is safe to run alongside the API: the service serializes
cache misses with a per-window lock and atomically replaces the JSONL file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from backend.app.services.frozen_explanation_service import (
    _ledger_path,
    build_local_explanation,
)


parser = argparse.ArgumentParser()
parser.add_argument("--window", required=True)
parser.add_argument("--project", help="One canonical project code to reconstruct.")
parser.add_argument("--all", action="store_true", help="Warm every latest frozen-ledger project for the window.")
parser.add_argument("--limit", type=int, help="Maximum projects to process (useful for a smoke test).")
args = parser.parse_args()

if bool(args.project) == bool(args.all):
    parser.error("provide exactly one of --project or --all")


def failure_bucket(reason: str) -> str:
    reason = reason.lower()
    if "source row" in reason or "source snapshot" in reason:
        return "missing_source"
    if "prediction mismatch" in reason or "reproduce" in reason:
        return "mismatch"
    return "error"


if args.project:
    ledger = pd.read_csv(_ledger_path(args.window), dtype={"canonical_project_id": str}, low_memory=False)
    rows = ledger[ledger.canonical_project_id.eq(str(args.project))].copy()
else:
    ledger = pd.read_csv(_ledger_path(args.window), dtype={"canonical_project_id": str}, low_memory=False)
    ledger["snapshot_date"] = pd.to_datetime(ledger["snapshot_date"], errors="coerce")
    rows = (ledger.sort_values("snapshot_date")
            .drop_duplicates("canonical_project_id", keep="last"))

if args.limit is not None:
    rows = rows.head(args.limit)

report = {"window": args.window, "total": int(len(rows)), "verified": 0,
          "mismatch": 0, "missing_source": 0, "error": 0, "projects": []}
for row in rows.itertuples(index=False):
    code = str(row.canonical_project_id)
    snapshot = pd.Timestamp(row.snapshot_date).strftime("%Y-%m-%d")
    try:
        entry = build_local_explanation(args.window, code, snapshot)
        report["verified"] += 1
        report["projects"].append({"project_code": code, "snapshot_date": snapshot, "status": "verified",
                                   "cache_identity": entry["cache_identity"]})
    except Exception as exc:
        reason = str(exc)
        bucket = failure_bucket(reason)
        report[bucket] += 1
        report["projects"].append({"project_code": code, "snapshot_date": snapshot, "status": bucket, "reason": reason})

print(json.dumps(report, indent=2, allow_nan=False))
