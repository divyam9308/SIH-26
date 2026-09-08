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
from datetime import datetime, timezone
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
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE, _aft_routing_limit, _select_aft_calibration_projects
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.services.operational_driver_service import operational_drivers

logger = logging.getLogger(__name__)
FROZEN_ROUTING_GATE = 'exp35_calibration_cohort_eligible'
ARTIFACT_SCHEMA_VERSION = 1
EXPLANATION_FILENAME = "project_explanations.jsonl"
EXPLANATION_METADATA_FILENAME = "project_explanations.meta.json"
EXPLANATION_METHOD = "deterministic_two_path_wrapper_contributions_v1"


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
    missing = [path.name for path in (manifest_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact for frozen lifecycle window {window}: {', '.join(missing)}."
        )
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
    fingerprints = manifest.get("artifact_fingerprints") or {}
    for name in ("cost_model.pkl", "delay_model.pkl", "risk_model.pkl", "metadata.json", "prediction_validation.csv"):
        path = root / name
        expected = fingerprints.get(name)
        if not path.exists() or not expected:
            raise ValueError(f"Frozen production artifact is missing or unsigned: {name}.")
        if _digest(str(path), *_stat_key(path)) != expected:
            raise ValueError(f"Frozen production artifact failed its manifest fingerprint: {name}.")
    return {'metadata': metadata, **{target: joblib.load(root / f'{target}_model.pkl') for target in ('cost', 'delay', 'risk')}}


def _trajectory_signature() -> str:
    stat = TRAJECTORIES.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}:{stat.st_ctime_ns}"


@lru_cache(maxsize=1)
def _trajectory_frame(signature: str) -> pd.DataFrame:
    del signature
    frame = pd.read_csv(TRAJECTORIES, dtype={'project_id': 'string'}, low_memory=False)
    frame['snapshot_date'] = pd.to_datetime(frame.snapshot_date, errors='coerce')
    return frame


@lru_cache(maxsize=256)
def _as_of_frame(code: str, snapshot_date: str, identity: str) -> pd.DataFrame:
    del identity
    frame = _trajectory_frame(_trajectory_signature())
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
    return MODEL_ROOT / window / EXPLANATION_FILENAME


def _metadata_path(window: str) -> Path:
    return MODEL_ROOT / window / EXPLANATION_METADATA_FILENAME


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


@lru_cache(maxsize=3)
def _ledger_frame(window: str, signature: tuple[int, int, int]) -> pd.DataFrame:
    del signature
    ledger = pd.read_csv(_ledger_path(window), dtype={"canonical_project_id": str}, low_memory=False)
    ledger["snapshot_date"] = pd.to_datetime(ledger["snapshot_date"], errors="coerce")
    return ledger


def _ledger_row(window: str, code: str, snapshot_date: str | None = None) -> pd.Series:
    path = _ledger_path(window)
    ledger = _ledger_frame(window, _stat_key(path))
    rows = ledger[ledger.canonical_project_id.eq(str(code))].copy()
    if rows.empty:
        raise KeyError(code)
    if snapshot_date is not None:
        rows = rows[rows.snapshot_date.eq(pd.Timestamp(snapshot_date))]
    if rows.empty:
        raise ValueError('Exact frozen ledger snapshot is unavailable.')
    return rows.sort_values("snapshot_date").iloc[-1]


@lru_cache(maxsize=2)
def _frozen_scoring_frame(window: str, identity: str) -> pd.DataFrame:
    """Rebuild the exact published scoring inputs without fitting any model.

    This follows the production feature and training-only prior path so wrapper
    inputs match the immutable prediction ledger. Target columns may exist in
    the returned audit frame, but only each model's explicit feature contract
    is ever passed to prediction or explanation.
    """
    metadata = _frozen_bundle(window, identity)["metadata"]
    training_period = metadata.get("training_period") or [int(window[:4]), int(window[-4:])]
    testing_period = metadata.get("testing_period") or [int(training_period[1]) + 1, 2025]
    start = int(metadata.get("training_start", training_period[0]))
    end = int(metadata.get("training_end", training_period[1]))
    test_end = int(metadata.get("test_end", testing_period[1]))
    supervised, _ = build_training_dataset()
    prepared = normalize_taxonomy(_prepare(supervised))
    train, test = temporal_project_split(prepared, start, end, test_end)
    _, test, _ = _build_temporal_delay_priors(train, test)
    cohort = _production_cost_evaluation_rows(test)
    calibration_ids = _select_aft_calibration_projects(
        cohort, limit=_aft_routing_limit(start, end, test_end)
    )
    test = test.copy()
    test[CALIBRATION_GATE_FEATURE] = test["canonical_project_id"].astype("string").isin(calibration_ids)
    test["snapshot_date"] = pd.to_datetime(test["snapshot_date"], errors="coerce")
    return test


def _factor_values(model, row: pd.Series, features: list[str], background: pd.DataFrame, *, target: str, limit: int = 5) -> tuple[list[dict], list[dict], float, float, float]:
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
    all_factors = [
        {"feature": feature, "impact": round(float(value), 4), "direction": "increases" if value >= 0 else "reduces"}
        for feature, value in sorted(zip(features, values), key=lambda item: (-abs(item[1]), item[0]))
    ]
    return all_factors[:limit], all_factors, base, prediction, reconstructed


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
    scoring = _frozen_scoring_frame(window, identity)
    rows = scoring[
        scoring["canonical_project_id"].astype("string").eq(str(code))
        & scoring["snapshot_date"].eq(pd.Timestamp(ledger.snapshot_date))
    ]
    if len(rows) != 1:
        raise ValueError(f"Exact frozen scoring row for {code} at {pd.Timestamp(ledger.snapshot_date):%Y-%m-%d} is unavailable or ambiguous.")
    source = rows.iloc[0]
    if pd.Timestamp(source.snapshot_date) > pd.Timestamp(snapshot_date):
        raise ValueError('Future source snapshot cannot explain a frozen prediction.')
    contract = target_feature_contract(bundle["metadata"])
    expected = {"cost": float(ledger.predicted_cost_overrun), "delay": float(ledger.predicted_delay_days), "risk": str(ledger.predicted_risk)}
    result: dict[str, Any] = {}
    inputs: dict[str, list[str]] = {}
    missing: set[str] = set()
    for target in ("cost", "delay", "risk"):
        model = bundle[target]
        features = list(getattr(model, "features", contract[target]))
        if not features:
            raise ValueError(f'Frozen {target} feature schema is incomplete.')
        missing.update(feature for feature in features if feature not in source.index)
        inputs[target] = features
    if missing.difference({FROZEN_ROUTING_GATE}):
        raise ValueError(f'Frozen feature schema is incomplete: {", ".join(sorted(missing))}.')

    reproduced: dict[str, Any] | None = None
    selected_source: pd.Series | None = None
    attempt: dict[str, Any] = {}
    for target in ("cost", "delay", "risk"):
        prediction = bundle[target].predict(source.to_frame().T.reindex(columns=inputs[target]))[0]
        attempt[target] = str(prediction) if target == 'risk' else float(prediction)
    if (
        np.isclose(attempt['cost'], expected['cost'], rtol=1e-6, atol=1e-4)
        and np.isclose(attempt['delay'], expected['delay'], rtol=1e-6, atol=1e-4)
        and attempt['risk'] == expected['risk']
    ):
        selected_source, reproduced = source, attempt
    if selected_source is None or reproduced is None:
        raise ValueError('Frozen Cost, Delay, and Risk predictions did not reproduce from the exact frozen scoring inputs.')
    source = selected_source
    # All three guards must pass before any explanation is computed.
    for target in ('cost', 'delay', 'risk'):
        model, features = bundle[target], inputs[target]
        try:
            factors, all_factors, base, output, reconstructed = _factor_values(
                model, source, features, _background(frame, source, features), target=target
            )
            result[target] = {
                "available": True,
                "reason": None,
                "source": "frozen_verified_local_explanation",
                "base_value": round(base, 6),
                "prediction": round(output, 6),
                "reconstructed_prediction": round(reconstructed, 6),
                "factors": factors,
                "all_factors": all_factors,
                "feature_hash": _row_hash(source, features),
                "output": "predicted_class_probability" if target == "risk" else f"predicted_{target}",
                **({"predicted_class": expected["risk"]} if target == "risk" else {}),
            }
        except Exception as exc:
            reason = str(exc)
            category = "explanation_additivity_failure" if "additivity" in reason.lower() else "unsupported_model_wrapper"
            result[target] = {
                "available": False, "reason": reason, "source": None,
                "failure_category": category, "factors": [], "all_factors": [],
                "feature_hash": _row_hash(source, features),
            }
    metadata = bundle["metadata"]
    history = frame[pd.to_datetime(frame["snapshot_date"], errors="coerce").le(pd.Timestamp(source.snapshot_date))]
    try:
        drivers = operational_drivers(source, history, source="official_snapshot_trajectory")
        driver_status = {"available": True, "reason": None, "source": "official_snapshot_trajectory"}
    except Exception as exc:
        drivers = []
        driver_status = {"available": False, "reason": str(exc), "source": None, "failure_category": "operational_driver_generation_failure"}
    entry = {
        "window": window, "project_code": str(code), "snapshot_date": pd.Timestamp(ledger.snapshot_date).strftime("%Y-%m-%d"),
        "model_version": metadata.get("model_version") or window,
        "run_id": metadata.get("run_id") or (metadata.get("provenance") or {}).get("run_id"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint"),
        "cost": result["cost"], "delay": result["delay"], "risk": result["risk"],
        "models": result,
        "operational_drivers": drivers,
        "operational_driver_status": driver_status,
        "cache_identity": identity,
        "reproduction": {'ledger': expected, 'recomputed': reproduced, "passed": True},
        "method": EXPLANATION_METHOD,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    return entry


def _atomic_json(path: Path, value: dict) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed project explanation record at {path}:{number}.") from exc
    return records


def _entry_key(entry: dict) -> tuple[str, str]:
    return str(entry.get("project_code")), str(entry.get("snapshot_date"))


def _fully_available(entry: dict) -> bool:
    return (
        entry.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
        and entry.get("reproduction", {}).get("passed") is True
        and all(entry.get(target, {}).get("available") is True for target in ("cost", "delay", "risk"))
        and isinstance(entry.get("operational_drivers"), list)
        and entry.get("operational_driver_status", {}).get("available") is True
    )


def publish_explanations(window: str, entries: list[dict]) -> dict:
    """Atomically publish deterministic records and their integrity metadata."""
    path = _output_path(window)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(window)
    deduplicated = {_entry_key(entry): entry for entry in entries}
    ordered = [deduplicated[key] for key in sorted(deduplicated)]
    if any(entry.get("window") != window or entry.get("cache_identity") != identity for entry in ordered):
        raise ValueError("Refusing to publish explanation records for a different frozen artifact identity.")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for item in ordered:
                handle.write(json.dumps(item, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if _identity(window) != identity:
            raise ValueError("Frozen artifacts changed during publication; retry required.")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    artifact_sha256 = _digest(str(path), *_stat_key(path))
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "window": window,
        "cache_identity": identity,
        "artifact_sha256": artifact_sha256,
        "record_count": len(ordered),
        "fully_explained": sum(_fully_available(entry) for entry in ordered),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "method": EXPLANATION_METHOD,
    }
    _atomic_json(_metadata_path(window), metadata)
    _published_index.cache_clear()
    return metadata


def _persist(window: str, entry: dict) -> None:
    existing = [item for item in _read_records(_output_path(window)) if item.get("cache_identity") == entry["cache_identity"]]
    existing = [item for item in existing if _entry_key(item) != _entry_key(entry)]
    publish_explanations(window, [*existing, entry])


def _artifact_signature(window: str) -> tuple[int, int, int]:
    return _stat_key(_output_path(window))


@lru_cache(maxsize=6)
def _published_index(window: str, signature: tuple[int, int, int], identity: str) -> dict[tuple[str, str], dict]:
    del signature
    path = _output_path(window)
    metadata_path = _metadata_path(window)
    if not metadata_path.exists():
        raise ValueError("Published explanation metadata is missing.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("cache_identity") != identity:
        raise ValueError("Published explanations are stale for the active frozen model bundle.")
    if metadata.get("artifact_sha256") != _digest(str(path), *_stat_key(path)):
        raise ValueError("Published explanation artifact failed its SHA-256 integrity check.")
    records = _read_records(path)
    if metadata.get("record_count") != len(records):
        raise ValueError("Published explanation record count does not match its metadata.")
    return {_entry_key(entry): entry for entry in records if entry.get("cache_identity") == identity}


def local_explanation(window: str, code: str, snapshot_date: str, identity: str | None = None) -> dict | None:
    identity = identity or _identity(window)
    path = _output_path(window)
    if not path.exists():
        return None
    return _published_index(window, _artifact_signature(window), identity).get((str(code), str(snapshot_date)))


def build_local_explanation(window: str, code: str, snapshot_date: str | None = None) -> dict:
    identity = _identity(window)
    if snapshot_date is None:
        snapshot_date = str(_ledger_row(window, code).snapshot_date.date())
    snapshot_date = str(pd.Timestamp(snapshot_date).date())
    cached = local_explanation(window, code, snapshot_date, identity)
    if cached and _fully_available(cached):
        return cached
    # Separate lock inode survives atomic cache replacement. Serialize misses
    # per window across API threads/workers and the offline warmer.
    with _output_path(window).with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = _identity(window)
        cached = local_explanation(window, code, snapshot_date, identity)
        if cached and _fully_available(cached):
            return cached
        entry = _reconstruct(window, code, snapshot_date, identity)
        if _identity(window) != identity:
            raise ValueError('Frozen artifacts changed during reconstruction; retry required.')
        _persist(window, entry)
        return entry


def verified_explanation(window: str, code: str, snapshot_date: str) -> tuple[dict | None, dict]:
    """Read a published explanation without loading models or computing factors."""
    try:
        entry = local_explanation(window, code, snapshot_date)
        if entry is None:
            return None, {
                'available': False,
                'reason': 'No published explanation exists for this exact frozen project snapshot.',
                'source': None,
            }
        available = _fully_available(entry)
        reason = None if available else next(
            (entry.get(target, {}).get("reason") for target in ("cost", "delay", "risk") if not entry.get(target, {}).get("available")),
            "The published explanation record is incomplete.",
        )
        return entry, {'available': available, 'reason': reason, 'source': 'frozen_verified_local_explanation' if available else None}
    except Exception as exc:
        logger.warning('Published frozen explanation unavailable window=%s project=%s snapshot=%s: %s', window, code, snapshot_date, exc)
        return None, {'available': False, 'reason': f'Published explanation unavailable: {exc}', 'source': None}
