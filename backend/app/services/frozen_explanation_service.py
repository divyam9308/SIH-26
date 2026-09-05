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
import logging
import fcntl
import re
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import joblib

from backend.app.ml.production_cost_baseline import target_feature_contract
from backend.app.services.monthly_prediction_service import MODEL_ROOT, TRAJECTORIES, _validate_bundle_provenance
from backend.app.ml.production_cost_baseline import enrich_history_for_production
from backend.app.ml.production_delay_baseline import enrich_history_for_delay_production

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _digest(path: str, mtime: int, size: int, ctime: int) -> str:
    del mtime, size, ctime
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _identity(window: str) -> str:
    if not re.fullmatch(r'\d{4}_\d{4}', window):
        raise ValueError('Invalid frozen window.')
    root = MODEL_ROOT / window
    manifest_path = root / 'run_manifest.json'
    metadata_path = root / 'metadata.json'
    manifest = json.loads(manifest_path.read_text())
    fingerprints = manifest.get('artifact_fingerprints') or {}
    required = ('cost_model.pkl', 'delay_model.pkl', 'risk_model.pkl', 'metadata.json', 'prediction_validation.csv')
    if not all(fingerprints.get(name) for name in required):
        raise ValueError('Frozen manifest is missing artifact fingerprints.')
    # The signed artifact fingerprints in the immutable manifest, plus the
    # exact manifest and metadata contents, make this portable across hosts.
    # That avoids re-hashing 500+ MB model files before a cache hit, while a
    # modified manifest or bundle necessarily changes the cache key.
    payload = {
        'schema': 'verified-cache-v3',
        'manifest': manifest,
        'metadata_sha256': _digest(str(metadata_path), *(_stat_key(metadata_path))),
        'manifest_sha256': _digest(str(manifest_path), *(_stat_key(manifest_path))),
        'artifact_fingerprints': {name: fingerprints[name] for name in required},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _stat_key(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns


@lru_cache(maxsize=3)
def _frozen_bundle(window: str, identity: str) -> dict:
    del identity
    root = MODEL_ROOT / window
    metadata = json.loads((root / 'metadata.json').read_text())
    manifest = json.loads((root / 'run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('model_role') != 'production':
        raise ValueError('Frozen bundle is not complete production evidence.')
    _validate_bundle_provenance(window, metadata, manifest)
    return {'metadata': metadata, **{target: joblib.load(root / f'{target}_model.pkl') for target in ('cost', 'delay', 'risk')}}


@lru_cache(maxsize=128)
def _as_of_frame(code: str, snapshot_date: str, identity: str) -> pd.DataFrame:
    del identity
    frame = pd.read_csv(TRAJECTORIES, dtype={'project_id': 'string'}, low_memory=False)
    frame['snapshot_date'] = pd.to_datetime(frame.snapshot_date, errors='coerce')
    # Filter to the requested source trajectory and its allowed time horizon
    # BEFORE either production history/aggregate feature builder runs.  The
    # agency/sector priors used by the frozen contract are source columns in
    # this trajectory; no other project's later observations are needed.
    frame = frame[
        frame.project_id.astype('string').str.upper().eq(str(code).upper())
        & frame.snapshot_date.le(pd.Timestamp(snapshot_date))
    ].copy()
    if frame.empty:
        raise ValueError(f'Frozen source row for {code} at {snapshot_date} is unavailable.')
    return enrich_history_for_delay_production(enrich_history_for_production(frame))


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


def _ledger_row(window: str, code: str, snapshot_date: str | None = None) -> pd.Series:
    ledger = pd.read_csv(_ledger_path(window), dtype={"canonical_project_id": str}, low_memory=False)
    rows = ledger[ledger.canonical_project_id.eq(str(code))].copy()
    if rows.empty:
        raise KeyError(code)
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    if snapshot_date is not None:
        rows = rows[rows.snapshot_date.eq(pd.Timestamp(snapshot_date))]
    if rows.empty:
        raise ValueError('Exact frozen ledger snapshot is unavailable.')
    return rows.sort_values("snapshot_date").iloc[-1]


def _source_row(code: str, snapshot_date: pd.Timestamp, frame: pd.DataFrame) -> pd.Series:
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
        states: list[pd.Series] = []
        for index in order:
            working.iloc[index] = row_frame.iloc[0, index]
            states.append(working.copy())
        # Each path state is independent, so a single batch prediction is
        # numerically equivalent to the prior row-at-a-time evaluation and
        # avoids hundreds of wrapper calls during a cache miss.
        outputs = predict(pd.DataFrame(states, columns=features))
        previous = base
        for index, current in zip(order, outputs):
            totals[index] += current - previous
            previous = float(current)
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


def _reconstruct(window: str, code: str, snapshot_date: str, identity: str) -> dict:
    bundle = _frozen_bundle(window, identity)
    ledger = _ledger_row(window, code, snapshot_date)
    frame = _as_of_frame(code, snapshot_date, identity)
    source = _source_row(code, pd.Timestamp(ledger.snapshot_date), frame)
    if pd.Timestamp(source.snapshot_date) > pd.Timestamp(snapshot_date):
        raise ValueError('Future source snapshot cannot explain a frozen prediction.')
    contract = target_feature_contract(bundle["metadata"])
    expected = {"cost": float(ledger.predicted_cost_overrun), "delay": float(ledger.predicted_delay_days), "risk": str(ledger.predicted_risk)}
    result: dict[str, Any] = {}
    inputs = {}
    reproduced = {}
    for target in ("cost", "delay", "risk"):
        model = bundle[target]
        features = list(getattr(model, "features", contract[target]))
        if not features or any(feature not in source.index for feature in features):
            raise ValueError(f'Frozen {target} feature schema is incomplete.')
        inputs[target] = features
        prediction = model.predict(source.to_frame().T.reindex(columns=features))[0]
        if target == "risk":
            if str(prediction) != expected[target]:
                raise ValueError(f"Frozen {target} prediction mismatch: {prediction!r} != {expected[target]!r}")
        elif not np.isclose(float(prediction), expected[target], rtol=1e-6, atol=1e-4):
            raise ValueError(f"Frozen {target} prediction mismatch: {float(prediction):.6f} != {expected[target]:.6f}")
        reproduced[target] = str(prediction) if target == 'risk' else float(prediction)
    # All three guards must pass before any explanation is computed.
    for target in ('cost', 'delay', 'risk'):
        model, features = bundle[target], inputs[target]
        factors, base, output = _factor_values(model, source, features, _background(frame, source, features), target=target)
        result[target] = {"factors": factors, "base_value": round(base, 6), "prediction": round(output, 6), "feature_hash": _row_hash(source, features)}
    metadata = bundle["metadata"]
    entry = {
        "window": window, "project_code": str(code), "snapshot_date": pd.Timestamp(ledger.snapshot_date).strftime("%Y-%m-%d"),
        "run_id": metadata.get("run_id") or (metadata.get("provenance") or {}).get("run_id"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint"),
        "models": result,
        "cache_identity": identity,
        "reproduction": {'ledger': expected, 'recomputed': reproduced},
        "method": 'two_antithetic_permutations_single_reference',
    }
    return entry


def _persist(window: str, entry: dict) -> None:
    path = _output_path(window)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError:
                # An incomplete line can only come from an externally edited
                # cache: our own writes use fsync + atomic replacement.  Do
                # not let that prevent a verified replacement entry.
                logger.warning("Ignoring malformed local-SHAP cache entry in %s", path)
    existing = [item for item in existing if not (item.get("project_code") == entry["project_code"] and item.get("snapshot_date") == entry["snapshot_date"])]
    existing.append(entry)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("\n".join(json.dumps(item, sort_keys=True, allow_nan=False) for item in existing) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def local_explanation(window: str, code: str, snapshot_date: str, identity: str | None = None) -> dict | None:
    identity = identity or _identity(window)
    path = _output_path(window)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed local-SHAP cache entry in %s", path)
            continue
        if (item.get('window') == window and item.get('cache_identity') == identity
                and item.get("project_code") == str(code) and item.get("snapshot_date") == str(snapshot_date)
                and all(item.get('models', {}).get(t, {}).get('factors') for t in ('cost', 'delay', 'risk'))):
            return item
    return None


def build_local_explanation(window: str, code: str, snapshot_date: str | None = None) -> dict:
    identity = _identity(window)
    if snapshot_date is None:
        snapshot_date = str(_ledger_row(window, code).snapshot_date.date())
    snapshot_date = str(pd.Timestamp(snapshot_date).date())
    cached = local_explanation(window, code, snapshot_date, identity)
    if cached:
        return cached
    # Separate lock inode survives atomic cache replacement. Serialize misses
    # per window across API threads/workers and the offline warmer.
    with _output_path(window).with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = _identity(window)
        cached = local_explanation(window, code, snapshot_date, identity)
        if cached:
            return cached
        entry = _reconstruct(window, code, snapshot_date, identity)
        if _identity(window) != identity:
            raise ValueError('Frozen artifacts changed during reconstruction; retry required.')
        _persist(window, entry)
        return entry


def verified_explanation(window: str, code: str, snapshot_date: str) -> tuple[dict | None, dict]:
    try:
        entry = build_local_explanation(window, code, snapshot_date)
        return entry, {'available': True, 'reason': None, 'source': 'verified_frozen_local_shap_ledger'}
    except Exception as exc:
        # Explanation failures must not break the frozen prediction endpoint.
        logger.warning('Frozen explanation unavailable window=%s project=%s snapshot=%s: %s', window, code, snapshot_date, exc, exc_info=True)
        return None, {'available': False, 'reason': f'Frozen explanation unavailable: {exc}', 'source': 'frozen_evaluation_ledger'}
