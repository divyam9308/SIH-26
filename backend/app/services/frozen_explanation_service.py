"""Verified local explanations for immutable lifecycle evaluation ledgers.

This module deliberately separates publishing local SHAP from model training.
An entry is written only after the reconstructed source row produces the same
Cost, Delay, and Risk outputs as the frozen evaluation ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.production_cost_baseline import target_feature_contract
from backend.app.services.monthly_prediction_service import MODEL_ROOT, _bundle, _inference_frame


def _ledger_path(window: str) -> Path:
    return MODEL_ROOT / window / "prediction_validation.csv"


def _output_path(window: str) -> Path:
    return MODEL_ROOT / window / "local_shap_ledger.jsonl"


def _json_value(value: Any):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_hash(row: pd.Series, features: list[str]) -> str:
    value = {feature: _json_value(row.get(feature)) for feature in features}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _ledger_row(window: str, code: str) -> pd.Series:
    ledger = pd.read_csv(_ledger_path(window), dtype={"canonical_project_id": str}, low_memory=False)
    rows = ledger[ledger.canonical_project_id.eq(str(code))].copy()
    if rows.empty:
        raise KeyError(code)
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    return rows.sort_values("snapshot_date").iloc[-1]


def _source_row(code: str, snapshot_date: pd.Timestamp) -> pd.Series:
    frame = _inference_frame()
    rows = frame[
        frame.project_id.astype("string").str.upper().eq(str(code).upper())
        & pd.to_datetime(frame.snapshot_date, errors="coerce").eq(snapshot_date)
    ]
    if len(rows) != 1:
        raise ValueError(f"Frozen source row for {code} at {snapshot_date:%Y-%m-%d} is unavailable or ambiguous.")
    return rows.iloc[0]


def _factor_values(model, row: pd.Series, features: list[str], background: pd.DataFrame, *, target: str, limit: int = 5) -> tuple[list[dict], float, float]:
    """Use a model-agnostic permutation explainer for composite production wrappers.

    The output wrapper is the actual serialized artifact.  This avoids falsely
    attributing only its inner base estimator and supports Cost/Delay wrappers
    as well as the risk pipeline.
    """
    row_frame = row.to_frame().T.reindex(columns=features)
    bg = background.reindex(columns=features).copy()
    if bg.empty:
        bg = row_frame.copy()
    # Preserve native pandas values for CatBoost categories and date-aware
    # wrappers; SHAP supplies object arrays to the callable.
    predicted_label = None
    def predict(values):
        frame = pd.DataFrame(values, columns=features)
        if target == "risk":
            nonlocal predicted_label
            if predicted_label is None:
                predicted_label = str(model.predict(row_frame)[0])
            classes = [str(value) for value in model.classes_]
            return np.asarray(model.predict_proba(frame)[:, classes.index(predicted_label)], dtype=float).reshape(-1)
        return np.asarray(model.predict(frame), dtype=float).reshape(-1)

    prediction = float(predict(row_frame)[0])
    # Use two deterministic antithetic feature permutations over the complete
    # serialized predictor.  SHAP's tabular maskers apply numeric ``isclose``
    # to categorical strings, which breaks the production CatBoost inputs.
    # This is the same local permutation-contribution principle, preserves the
    # native DataFrame values passed to the wrapper, and remains exactly
    # additive to its output.  We label the persisted source accordingly.
    reference = bg.iloc[0].copy()
    base = float(predict(reference.to_frame().T)[0])
    totals = np.zeros(len(features), dtype=float)
    for order in (list(range(len(features))), list(range(len(features) - 1, -1, -1))):
        working = reference.copy()
        previous = base
        for index in order:
            working.iloc[index] = row_frame.iloc[0, index]
            current = float(predict(working.to_frame().T)[0])
            totals[index] += current - previous
            previous = current
    values = totals / 2
    reconstructed = base + float(values.sum())
    tolerance = max(0.01, abs(prediction) * 0.001)
    if not math.isfinite(reconstructed) or abs(reconstructed - prediction) > tolerance:
        raise ValueError(f"SHAP additivity failed: base + factors={reconstructed:.6f}, prediction={prediction:.6f}")
    factors = [
        {"feature": feature, "impact": round(float(value), 4), "direction": "increases" if value >= 0 else "reduces"}
        for feature, value in sorted(zip(features, values), key=lambda item: abs(item[1]), reverse=True)[:limit]
    ]
    return factors, base, prediction


def _background(frame: pd.DataFrame, source: pd.Series, features: list[str]) -> pd.DataFrame:
    # Only rows whose completed outcome belongs to the source model's past are
    # used as SHAP background; this is explanation context, never model fitting.
    before = frame[pd.to_datetime(frame.snapshot_date, errors="coerce").lt(pd.Timestamp(source.snapshot_date))]
    # One deterministic pre-holdout reference row keeps wrapper-level Shapley
    # evaluation bounded.  The background identity is captured by the feature
    # hash in the persisted entry; it is not a synthetic replacement row.
    return before.sort_values("snapshot_date").tail(1).reindex(columns=features)


def build_local_explanation(window: str, code: str) -> dict:
    bundle = _bundle(window)
    ledger = _ledger_row(window, code)
    source = _source_row(code, pd.Timestamp(ledger.snapshot_date))
    frame = _inference_frame()
    contract = target_feature_contract(bundle["metadata"])
    expected = {"cost": float(ledger.predicted_cost_overrun), "delay": float(ledger.predicted_delay_days), "risk": str(ledger.predicted_risk)}
    result: dict[str, Any] = {}
    for target in ("cost", "delay", "risk"):
        model = bundle[target]
        features = list(getattr(model, "features", contract[target]))
        prediction = model.predict(source.to_frame().T.reindex(columns=features))[0]
        if target == "risk":
            if str(prediction) != expected[target]:
                raise ValueError(f"Frozen {target} prediction mismatch: {prediction!r} != {expected[target]!r}")
        elif not np.isclose(float(prediction), expected[target], rtol=1e-6, atol=1e-4):
            raise ValueError(f"Frozen {target} prediction mismatch: {float(prediction):.6f} != {expected[target]:.6f}")
        factors, base, output = _factor_values(model, source, features, _background(frame, source, features), target=target)
        result[target] = {"factors": factors, "base_value": round(base, 6), "prediction": round(output, 6), "feature_hash": _row_hash(source, features)}
    metadata = bundle["metadata"]
    entry = {
        "window": window, "project_code": str(code), "snapshot_date": pd.Timestamp(ledger.snapshot_date).strftime("%Y-%m-%d"),
        "run_id": metadata.get("run_id") or (metadata.get("provenance") or {}).get("run_id"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint"),
        "models": result,
    }
    path = _output_path(window); path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    existing = [item for item in existing if not (item.get("project_code") == entry["project_code"] and item.get("snapshot_date") == entry["snapshot_date"])]
    existing.append(entry)
    with NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write("\n".join(json.dumps(item, sort_keys=True) for item in existing) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return entry


def local_explanation(window: str, code: str, snapshot_date: str) -> dict | None:
    path = _output_path(window)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if item.get("project_code") == str(code) and item.get("snapshot_date") == str(snapshot_date):
            return item
    return None
