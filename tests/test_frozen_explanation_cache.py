from __future__ import annotations

import math
import numpy as np
import pandas as pd

from backend.app.services import frozen_explanation_service as explanations


def _entry(identity: str) -> dict:
    factors = [{"feature": "expenditure_ratio", "impact": 1.0, "direction": "increases"}]
    return {
        "window": "2001_2021", "project_code": "P1", "snapshot_date": "2023-05-31",
        "cache_identity": identity,
        "artifact_schema_version": explanations.ARTIFACT_SCHEMA_VERSION,
        "reproduction": {"passed": True},
        "operational_drivers": [],
        "operational_driver_status": {"available": True},
        **{target: {"available": True, "factors": factors} for target in ("cost", "delay", "risk")},
    }


def test_cache_key_includes_frozen_artifact_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(explanations, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(explanations, "_identity", lambda _window: "bundle-a")
    explanations._published_index.cache_clear()
    explanations._persist("2001_2021", _entry("bundle-a"))

    assert explanations.local_explanation("2001_2021", "P1", "2023-05-31", "bundle-a")
    try:
        explanations.local_explanation("2001_2021", "P1", "2023-05-31", "bundle-b")
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("A stale artifact identity must be rejected explicitly")


def test_as_of_reconstruction_filters_before_feature_enrichment(monkeypatch, tmp_path):
    source = tmp_path / "trajectories.csv"
    pd.DataFrame([
        {"project_id": "P1", "snapshot_date": "2023-05-31"},
        {"project_id": "P1", "snapshot_date": "2023-06-30"},
    ]).to_csv(source, index=False)
    monkeypatch.setattr(explanations, "TRAJECTORIES", source)
    monkeypatch.setattr(explanations, "enrich_history_for_production", lambda frame: frame)
    monkeypatch.setattr(explanations, "enrich_history_for_delay_production", lambda frame: frame)
    explanations._as_of_frame.cache_clear()

    reconstructed = explanations._as_of_frame("P1", "2023-05-31", "bundle-a")
    assert reconstructed["snapshot_date"].max() == pd.Timestamp("2023-05-31")


def test_api_lookup_never_builds_a_missing_explanation(monkeypatch):
    monkeypatch.setattr(explanations, "local_explanation", lambda *_args: None)
    monkeypatch.setattr(explanations, "build_local_explanation", lambda *_args: (_ for _ in ()).throw(AssertionError("must not build")))

    entry, status = explanations.verified_explanation("2001_2021", "P1", "2023-05-31")
    assert entry is None
    assert status["available"] is False
    assert "No published explanation" in status["reason"]


class _Regressor:
    def predict(self, frame):
        return np.asarray(frame["a"], dtype=float) * 2 + np.asarray(frame["b"], dtype=float) * 3 + 7


class _Classifier:
    classes_ = np.array(["LOW", "HIGH"])

    def predict(self, frame):
        return np.where(np.asarray(frame["a"], dtype=float) >= 1, "HIGH", "LOW")

    def predict_proba(self, frame):
        high = np.clip(np.asarray(frame["a"], dtype=float) / 4, 0, 1)
        return np.column_stack([1 - high, high])


def test_wrapper_contributions_reconstruct_exact_final_numeric_output():
    row = pd.Series({"a": 2.0, "b": 5.0})
    background = pd.DataFrame([{"a": 0.0, "b": 1.0}])
    factors, all_factors, base, prediction, reconstructed = explanations._factor_values(
        _Regressor(), row, ["a", "b"], background, target="cost"
    )
    assert len(factors) <= 5
    assert all(math.isfinite(item["impact"]) for item in all_factors)
    assert math.isclose(base + sum(item["impact"] for item in all_factors), prediction, abs_tol=1e-4)
    assert math.isclose(reconstructed, prediction)


def test_risk_contributions_explain_predicted_class_probability():
    row = pd.Series({"a": 2.0, "b": 5.0})
    background = pd.DataFrame([{"a": 0.0, "b": 1.0}])
    _, all_factors, base, prediction, reconstructed = explanations._factor_values(
        _Classifier(), row, ["a", "b"], background, target="risk"
    )
    assert prediction == 0.5
    assert math.isclose(base + sum(item["impact"] for item in all_factors), prediction, abs_tol=1e-4)
    assert math.isclose(reconstructed, prediction)


def test_published_index_is_reused_without_rescanning(monkeypatch, tmp_path):
    monkeypatch.setattr(explanations, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(explanations, "_identity", lambda _window: "bundle-a")
    explanations._published_index.cache_clear()
    explanations._persist("2001_2021", _entry("bundle-a"))
    assert explanations.local_explanation("2001_2021", "P1", "2023-05-31", "bundle-a")
    monkeypatch.setattr(explanations, "_read_records", lambda *_args: (_ for _ in ()).throw(AssertionError("rescanned")))
    assert explanations.local_explanation("2001_2021", "P1", "2023-05-31", "bundle-a")
