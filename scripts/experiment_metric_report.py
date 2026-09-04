from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_PATH = Path("models/monthly_lifecycle/2001_2021/evaluation_results.json")
WINDOW = {"training_start": 2001, "training_end": 2021, "test_start": 2022, "test_end": 2025}


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    lifecycle = payload.get("lifecycle") or {}
    metrics = lifecycle.get("metrics") or {}
    cost = metrics.get("cost") or {}
    delay = metrics.get("delay") or {}
    risk = metrics.get("risk") or payload.get("risk_model") or {}
    def number(mapping: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        raise KeyError(f"Missing metric: {keys}")
    return {
        "cost_r2": number(cost, "R2", "r2"),
        "cost_mae": number(cost, "MAE", "mae"),
        "cost_rmse": number(cost, "RMSE", "rmse"),
        "delay_r2": number(delay, "R2", "r2"),
        "delay_mae": number(delay, "MAE", "MAE_days", "mae"),
        "delay_rmse": number(delay, "RMSE", "rmse"),
        "risk_macro_f1": number(risk, "macro_f1", "f1"),
        "risk_macro_precision": number(risk, "macro_precision", "precision"),
        "risk_macro_recall": number(risk, "macro_recall", "recall"),
    }


def _provenance(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    provenance = metadata.get("provenance") or {}
    training = metadata.get("training_period") or [metadata.get("training_start"), metadata.get("training_end")]
    testing = metadata.get("testing_period") or [metadata.get("test_start"), metadata.get("test_end")]
    return {
        "model_version": metadata.get("model_version"),
        "training_period": training,
        "testing_period": testing,
        "run_id": metadata.get("run_id"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint") or provenance.get("dataset_fingerprint"),
        "feature_schema_fingerprint": provenance.get("feature_schema_fingerprint"),
    }


def _assert_same_context(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    b = _provenance(base); c = _provenance(candidate)
    if list(b.get("training_period") or []) != [2001, 2021] or list(c.get("training_period") or []) != [2001, 2021]:
        raise AssertionError(f"Training window changed: baseline={b['training_period']} candidate={c['training_period']}")
    if list(b.get("testing_period") or []) != [2022, 2025] or list(c.get("testing_period") or []) != [2022, 2025]:
        raise AssertionError(f"Frozen holdout changed: baseline={b['testing_period']} candidate={c['testing_period']}")
    if b.get("dataset_fingerprint") and c.get("dataset_fingerprint") and b["dataset_fingerprint"] != c["dataset_fingerprint"]:
        raise AssertionError("Dataset fingerprint changed; metric comparison is not admissible")


def build_report(*, baseline: dict[str, Any], candidate: dict[str, Any], target: str) -> dict[str, Any]:
    _assert_same_context(baseline, candidate)
    b = _metrics(baseline); c = _metrics(candidate)
    delta = {key: c[key] - b[key] for key in b}
    if target == "cost_r2":
        accepted = c["cost_r2"] > b["cost_r2"] and c["cost_mae"] <= b["cost_mae"] * 1.01
        reason = "Cost R2 improved and Cost MAE stayed within the 1% non-regression gate" if accepted else "Cost R2 did not improve, or Cost MAE regressed by more than 1%"
    elif target == "delay_r2":
        accepted = c["delay_r2"] > b["delay_r2"] and c["delay_mae"] <= b["delay_mae"] * 1.01
        reason = "Delay R2 improved and Delay MAE stayed within the 1% non-regression gate" if accepted else "Delay R2 did not improve, or Delay MAE regressed by more than 1%"
    elif target == "risk_metrics":
        accepted = all(c[k] > b[k] for k in ("risk_macro_f1", "risk_macro_precision", "risk_macro_recall"))
        reason = "Macro F1, precision, and recall all improved" if accepted else "At least one of macro F1, precision, or recall did not improve"
    else:
        raise ValueError(target)
    return {
        "target": target,
        "decision": "ACCEPT" if accepted else "REJECT",
        "reason": reason,
        "window": WINDOW,
        "baseline": {"metrics": b, "provenance": _provenance(baseline)},
        "candidate": {"metrics": c, "provenance": _provenance(candidate)},
        "delta_candidate_minus_baseline": delta,
        "holdout_used_for_selection": False,
    }


def load_frozen_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen 2001-2021 production evaluation is required at {path}. "
            "Restore the canonical local production artifact; do not substitute legacy 2001_2022 data."
        )
    return json.loads(path.read_text())


def write_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
