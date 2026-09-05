"""Validate Projects UI cohorts against canonical published explanations."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services import frozen_explanation_service as explanations
from backend.app.services.range_portfolio_service import portfolio_payload

EXPLANATION_WINDOWS = frozenset({"2001_2021", "2001_2022"})

def verify_window(window: str) -> dict:
    expected_rows = portfolio_payload(window)["items"]
    expected = {(str(row["project_code"]), str(row["snapshot_date"])[:10]): row for row in expected_rows}
    report = {
        "projects": len(expected), "explanation_records": 0,
        "cost_explanations": 0, "delay_explanations": 0, "risk_explanations": 0,
        "operational_driver_records": 0, "fully_populated": 0,
        "missing": [], "stale": [], "prediction_reproduction_failures": [], "failures": [],
    }
    try:
        identity = explanations._identity(window)
        path = explanations._output_path(window)
        index = explanations._published_index(window, explanations._artifact_signature(window), identity)
        report["explanation_records"] = len(index)
    except Exception as exc:
        report["failures"].append({"category": "artifact_unavailable", "reason": str(exc)})
        return report

    expected_model_version = None
    expected_dataset = None
    if expected_rows:
        expected_model_version = expected_rows[0].get("model_version")
    metadata = json.loads((explanations.MODEL_ROOT / window / "metadata.json").read_text())
    expected_dataset = metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint")
    for key, project in expected.items():
        entry = index.get(key)
        if entry is None:
            report["missing"].append({"project_code": key[0], "snapshot_date": key[1]})
            continue
        stale_reasons = []
        if entry.get("window") != window or entry.get("snapshot_date") != key[1]:
            stale_reasons.append("window_or_snapshot_mismatch")
        if entry.get("model_version") != expected_model_version:
            stale_reasons.append("model_version_mismatch")
        if entry.get("dataset_fingerprint") != expected_dataset:
            stale_reasons.append("dataset_fingerprint_mismatch")
        if entry.get("cache_identity") != identity:
            stale_reasons.append("cache_identity_mismatch")
        if stale_reasons:
            report["stale"].append({"project_code": key[0], "snapshot_date": key[1], "reasons": stale_reasons})
        for target in ("cost", "delay", "risk"):
            target_record = entry.get(target, {})
            report[f"{target}_explanations"] += int(
                target_record.get("available") is True
                and isinstance(target_record.get("factors"), list)
                and all(isinstance(factor.get("impact"), (int, float)) for factor in target_record.get("factors", []))
            )
        report["operational_driver_records"] += int(
            isinstance(entry.get("operational_drivers"), list)
            and entry.get("operational_driver_status", {}).get("available") is True
        )
        if entry.get("reproduction", {}).get("passed") is not True:
            report["prediction_reproduction_failures"].append({
                "project_code": key[0], "snapshot_date": key[1],
                "reproduction": entry.get("reproduction"),
            })
        report["fully_populated"] += int(explanations._fully_available(entry) and not stale_reasons)
        for target in ("cost", "delay", "risk"):
            if entry.get(target, {}).get("available") is not True:
                report["failures"].append({
                    "project_code": key[0], "snapshot_date": key[1], "target": target,
                    "category": entry.get(target, {}).get("failure_category", "unavailable"),
                    "reason": entry.get(target, {}).get("reason"),
                })
    extras = sorted(set(index) - set(expected))
    if extras:
        report["failures"].append({"category": "unexpected_records", "projects": [list(key) for key in extras]})
    return report


def passed(report: dict) -> bool:
    projects = report["projects"]
    return (
        report["fully_populated"] == projects
        and report["explanation_records"] == projects
        and not report["missing"] and not report["stale"]
        and not report["prediction_reproduction_failures"] and not report["failures"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="append", choices=sorted(EXPLANATION_WINDOWS))
    parser.add_argument("--report", type=Path, default=Path("test-output/project-explanation-coverage.json"))
    args = parser.parse_args()
    windows = args.window or sorted(EXPLANATION_WINDOWS)
    document = {"generated_at": datetime.now(timezone.utc).isoformat(), "windows": {}}
    success = True
    for window in windows:
        report = verify_window(window)
        document["windows"][window] = report
        window_passed = passed(report)
        success = success and window_passed
        projects = report["projects"]
        percent = lambda count: 0 if not projects else round(count / projects * 100, 2)
        print(
            f"Window: {window}\nExpected projects: {projects}\n"
            f"Explanation records: {report['explanation_records']}\n"
            f"Cost coverage: {percent(report['cost_explanations'])}%\n"
            f"Delay coverage: {percent(report['delay_explanations'])}%\n"
            f"Risk coverage: {percent(report['risk_explanations'])}%\n"
            f"Operational-driver coverage: {percent(report['operational_driver_records'])}%\n"
            f"Missing: {len(report['missing'])}\nStale: {len(report['stale'])}\n"
            f"Prediction reproduction failures: {len(report['prediction_reproduction_failures'])}\n"
            f"{'PASS' if window_passed else 'FAIL'}\n"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
