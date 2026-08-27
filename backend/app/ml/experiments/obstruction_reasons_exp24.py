"""Experiment 24: delay-reason / obstruction text extraction coverage gate.

This experiment refuses to manufacture obstruction labels from numeric status
fields. It first proves that historical, prediction-time reason text is present
in the supervised PAIMANA evidence. If the repository does not contain enough
such text for both training windows, CI reports NOT EVALUABLE with a green
scientific workflow and leaves production unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract, train_window_with_promoted_cost

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "reports" / "experiments" / "exp_24"
RAW_ARCHIVE = ROOT / "data" / "raw" / "paimana_archive"
EXPERIMENT_ID = "exp_24"
EXPERIMENT_NAME = "As-of obstruction / delay-reason text features"
EXPERIMENT_SCOPE = "cost_delay"
REASON_COLUMNS = [
    "delay_reason", "reason_for_delay", "obstruction_reason", "obstruction",
    "constraints", "issues", "remarks", "project_remarks", "delay_remarks",
]
MIN_TRAIN_PROJECTS_WITH_REASON = 100

CATEGORY_PATTERNS = {
    "land_acquisition": r"land acquisition|acquisition of land|right of way|\brow\b",
    "clearance": r"environment|forest clearance|wildlife|clearance|permission",
    "utility_shifting": r"utility shifting|shifting of utilit|power line shifting|water line shifting",
    "contractor": r"contractor|mobilisation|mobilization|poor progress|slow progress",
    "funding": r"fund|finance|financial closure|payment|cash flow",
    "litigation": r"court|litigation|arbitration|legal dispute",
    "law_order": r"law and order|local agitation|protest|security issue",
    "geology_weather": r"geolog|landslide|flood|rain|weather|terrain",
    "procurement": r"tender|procurement|bid|award of work|contract award",
    "rehabilitation": r"rehabilitation|resettlement|r&r|compensation",
}


def categorize_obstruction_reason(value: object) -> dict[str, float]:
    text = "" if value is None or pd.isna(value) else str(value).lower()
    return {name: float(bool(re.search(pattern, text, flags=re.I))) for name, pattern in CATEGORY_PATTERNS.items()}


def reason_text(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    present = [column for column in REASON_COLUMNS if column in frame.columns]
    if not present:
        return pd.Series("", index=frame.index, dtype="string"), []
    text = frame[present].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    return text.astype("string"), present


def _raw_pdf_audit() -> dict:
    pdfs = sorted(RAW_ARCHIVE.glob("*.pdf")) if RAW_ARCHIVE.exists() else []
    return {
        "tracked_pdf_count": len(pdfs),
        "tracked_pdf_names": [path.name for path in pdfs[:30]],
        "note": "Tracked PDFs alone are not accepted as training evidence unless project-level as-of reason text can be joined to supervised snapshots.",
    }


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    data, identity = build_training_dataset()
    temp_root = Path(tempfile.mkdtemp(prefix="sih-exp24-"))
    run_id = f"exp24-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    try:
        production = train_window_with_promoted_cost(training_start, training_end, test_end, data=data, identity=identity, artifact_root=temp_root)
        artifact_dir = temp_root / f"{training_start}_{training_end}"
        metadata = production["metadata"]
        contract = target_feature_contract(metadata)

        enriched = enrich_supervised_for_production(data.copy())
        enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
        train, test = temporal_project_split(enriched, training_start, training_end, test_end)
        prod_cost = joblib.load(artifact_dir / "cost_model.pkl")
        prod_delay = joblib.load(artifact_dir / "delay_model.pkl")
        base_cost_pred = prod_cost.predict(test[contract["cost"]])
        base_delay_pred = np.maximum(0, prod_delay.predict(test[contract["delay"]]))
        base_cost = _regression_metrics(test.actual_cost_overrun_percentage, base_cost_pred, test.sample_weight, test.canonical_project_id)
        base_delay = _regression_metrics(test.actual_delay_days, base_delay_pred, test.sample_weight, test.canonical_project_id)

        train_text, columns = reason_text(train)
        test_text, _ = reason_text(test)
        train_has = train_text.str.len().gt(0)
        test_has = test_text.str.len().gt(0)
        training_projects_with_reason = int(train.loc[train_has, "canonical_project_id"].nunique())
        evaluable = bool(columns) and training_projects_with_reason >= MIN_TRAIN_PROJECTS_WITH_REASON

        # Current repository evidence is expected to fail this gate. If future
        # ingestion adds real as-of reason columns, this PR deliberately requires
        # a follow-up implementation/review before model fitting rather than
        # silently changing the scientific contract.
        report = {
            "experiment": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "status": "complete", "evaluable": False if not evaluable else False,
            "decision": "REGRESSION / DO NOT PROMOTE",
            "scientific_status": "NOT EVALUABLE - INSUFFICIENT HISTORICAL AS-OF REASON TEXT" if not evaluable else "REASON TEXT FOUND - MODEL IMPLEMENTATION REQUIRES REVIEW",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_period": [training_start, training_end], "testing_period": [training_end + 1, test_end],
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "metrics": {
                "production_cost_mae": float(base_cost["MAE"]), "experiment_cost_mae": None, "cost_improvement_percentage": None,
                "production_delay_mae": float(base_delay["MAE"]), "experiment_delay_mae": None, "delay_improvement_percentage": None,
            },
            "coverage": {
                "candidate_reason_columns_present": columns,
                "training_reason_snapshot_share": float(train_has.mean()),
                "test_reason_snapshot_share": float(test_has.mean()),
                "training_projects_with_reason": training_projects_with_reason,
                "minimum_training_projects_required": MIN_TRAIN_PROJECTS_WITH_REASON,
                "training_projects": int(train.canonical_project_id.nunique()),
                "test_projects": int(test.canonical_project_id.nunique()), "test_snapshots": int(len(test)),
                "raw_archive": _raw_pdf_audit(),
            },
            "reason_categories_ready_for_future_data": list(CATEGORY_PATTERNS),
            "leakage_policy": "Only project-level obstruction text explicitly present by the snapshot date is eligible. Final completion narratives, later reports, and numeric schedule-status proxies are forbidden as substitutes.",
            "production_changed": False,
        }
        out = REPORT_ROOT / f"{training_start}_{training_end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
