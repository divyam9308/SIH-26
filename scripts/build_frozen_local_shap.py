"""Publish verified project explanations from existing production artifacts.

This command never trains or modifies a model. Work is journaled separately;
the last verified canonical artifact remains intact until atomic publication.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Some persisted sklearn estimators emit this once per tree evaluation. It is
# an upstream parallelism advisory, not a data or explanation failure, and can
# otherwise produce gigabytes of duplicate publisher output.
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used.*",
    category=UserWarning,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services import frozen_explanation_service as explanations
from backend.app.services.range_portfolio_service import portfolio_payload

EXPLANATION_WINDOWS = frozenset({"2001_2021", "2001_2022"})

FAILURE_PATTERNS = (
    ("ambiguous", "ambiguous_source_snapshot"),
    ("source row", "missing_source_snapshot"),
    ("source snapshot", "missing_source_snapshot"),
    ("reproduce", "prediction_reproduction_failure"),
    ("artifact", "missing_model_artifact"),
    ("feature schema", "feature_schema_mismatch"),
    ("additivity", "explanation_additivity_failure"),
    ("wrapper", "unsupported_model_wrapper"),
)


def failure_category(reason: str) -> str:
    lowered = reason.lower()
    return next((category for needle, category in FAILURE_PATTERNS if needle in lowered), "explanation_generation_failure")


def cohort(window: str) -> list[tuple[str, str]]:
    rows = portfolio_payload(window)["items"]
    return sorted((str(row["project_code"]), str(row["snapshot_date"])[:10]) for row in rows)


def append_checkpoint(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def unavailable_entry(window: str, code: str, snapshot: str, identity: str, reason: str) -> dict:
    category = failure_category(reason)
    status = {"available": False, "reason": reason, "source": None, "failure_category": category, "factors": [], "all_factors": []}
    return {
        "window": window, "project_code": code, "snapshot_date": snapshot,
        "model_version": window, "run_id": None, "dataset_fingerprint": None,
        "cache_identity": identity, "artifact_schema_version": explanations.ARTIFACT_SCHEMA_VERSION,
        "cost": dict(status), "delay": dict(status), "risk": dict(status),
        "models": {target: dict(status) for target in ("cost", "delay", "risk")},
        "operational_drivers": [],
        "operational_driver_status": {"available": False, "reason": reason, "source": None, "failure_category": category},
        "reproduction": {"ledger": {}, "recomputed": {}, "passed": False},
        "method": explanations.EXPLANATION_METHOD,
    }


def _build_window_unlocked(window: str, args: argparse.Namespace) -> dict:
    rows = cohort(window)
    if args.project:
        rows = [row for row in rows if row[0] == str(args.project)]
        if not rows:
            raise ValueError(f"Project {args.project} is not in the {window} Projects UI cohort.")
    if args.limit is not None:
        rows = rows[:args.limit]
    identity = explanations._identity(window)
    output_path = explanations._output_path(window)
    checkpoint = output_path.with_suffix(".building.jsonl")
    existing = explanations._read_records(output_path) if output_path.exists() else []
    records = {
        explanations._entry_key(entry): entry for entry in existing
        if entry.get("cache_identity") == identity
    }
    if args.resume and checkpoint.exists():
        for entry in explanations._read_records(checkpoint):
            if entry.get("cache_identity") == identity:
                records[explanations._entry_key(entry)] = entry
    elif checkpoint.exists():
        checkpoint.unlink()

    report = {
        "window": window, "projects": len(rows), "cost_explanations": 0,
        "delay_explanations": 0, "risk_explanations": 0,
        "operational_driver_records": 0, "fully_populated": 0,
        "skipped_valid": 0, "failures": [],
    }
    print(f"Window: {window}\nProjects in UI cohort: {len(rows)}", flush=True)
    for number, (code, snapshot) in enumerate(rows, start=1):
        key = (code, snapshot)
        cached = records.get(key)
        if cached and explanations._fully_available(cached) and not args.force:
            report["skipped_valid"] += 1
            continue
        print(f"[{number}/{len(rows)}] {code} @ {snapshot}", flush=True)
        try:
            entry = explanations._reconstruct(window, code, snapshot, identity)
            for target in ("cost", "delay", "risk"):
                if entry[target].get("available") is not True:
                    report["failures"].append({
                        "project_code": code, "snapshot_date": snapshot, "target": target,
                        "category": entry[target].get("failure_category", "explanation_generation_failure"),
                        "reason": entry[target].get("reason"),
                    })
            if entry["operational_driver_status"].get("available") is not True:
                report["failures"].append({
                    "project_code": code, "snapshot_date": snapshot, "target": "operational_drivers",
                    "category": "operational_driver_generation_failure",
                    "reason": entry["operational_driver_status"].get("reason"),
                })
        except Exception as exc:
            reason = str(exc)
            entry = unavailable_entry(window, code, snapshot, identity, reason)
            report["failures"].append({"project_code": code, "snapshot_date": snapshot, "category": failure_category(reason), "reason": reason})
        records[key] = entry
        append_checkpoint(checkpoint, entry)

    selected = [records[row] for row in rows if row in records]
    published = selected if args.all else list(records.values())
    metadata = explanations.publish_explanations(window, published)
    if checkpoint.exists():
        checkpoint.unlink()
    selected_by_key = {explanations._entry_key(entry): entry for entry in selected}
    for key in rows:
        entry = selected_by_key.get(key, {})
        for target in ("cost", "delay", "risk"):
            report[f"{target}_explanations"] += int(entry.get(target, {}).get("available") is True)
        report["operational_driver_records"] += int(
            isinstance(entry.get("operational_drivers"), list)
            and entry.get("operational_driver_status", {}).get("available") is True
        )
        report["fully_populated"] += int(explanations._fully_available(entry))
    report["artifact"] = str(output_path)
    report["artifact_sha256"] = metadata["artifact_sha256"]
    print(
        f"Cost explained: {report['cost_explanations']}/{len(rows)}\n"
        f"Delay explained: {report['delay_explanations']}/{len(rows)}\n"
        f"Risk explained: {report['risk_explanations']}/{len(rows)}\n"
        f"Operational drivers evaluated: {report['operational_driver_records']}/{len(rows)}\n"
        f"Fully explained projects: {report['fully_populated']}/{len(rows)}\n"
        f"Verification failures: {len(report['failures'])}", flush=True,
    )
    return report


def build_window(window: str, args: argparse.Namespace) -> dict:
    lock_path = explanations._output_path(window).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_window_unlocked(window, args)


def main() -> int:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--window", choices=sorted(EXPLANATION_WINDOWS))
    selection.add_argument("--all-windows", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--project", help="One canonical project code to publish.")
    mode.add_argument("--all", action="store_true", help="Publish every Projects UI project.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.all_windows and not (args.project or args.all):
        parser.error("--window requires either --project or --all")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.all_windows:
        args.all = True

    aggregate = {"generated_at": datetime.now(timezone.utc).isoformat(), "windows": {}, "failures": []}
    for window in (sorted(EXPLANATION_WINDOWS) if args.all_windows else [args.window]):
        try:
            aggregate["windows"][window] = build_window(window, args)
        except Exception as exc:
            failure = {"window": window, "category": failure_category(str(exc)), "reason": str(exc)}
            aggregate["failures"].append(failure)
            print(f"Window {window}: FAILED — {exc}", file=sys.stderr, flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    print(json.dumps(aggregate, indent=2, allow_nan=False))
    return 1 if aggregate["failures"] or any(value["failures"] for value in aggregate["windows"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
