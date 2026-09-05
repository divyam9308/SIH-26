from __future__ import annotations

import pandas as pd

from backend.app.services import frozen_explanation_service as explanations


def _entry(identity: str) -> dict:
    factors = [{"feature": "expenditure_ratio", "impact": 1.0, "direction": "increases"}]
    return {
        "window": "2001_2021", "project_code": "P1", "snapshot_date": "2023-05-31",
        "cache_identity": identity,
        "models": {target: {"factors": factors} for target in ("cost", "delay", "risk")},
    }


def test_cache_key_includes_frozen_artifact_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(explanations, "MODEL_ROOT", tmp_path)
    explanations._persist("2001_2021", _entry("bundle-a"))

    assert explanations.local_explanation("2001_2021", "P1", "2023-05-31", "bundle-a")
    assert explanations.local_explanation("2001_2021", "P1", "2023-05-31", "bundle-b") is None


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


def test_failed_reproduction_is_explicitly_unavailable_and_not_cached(monkeypatch):
    monkeypatch.setattr(
        explanations, "build_local_explanation",
        lambda *_args: (_ for _ in ()).throw(ValueError("Frozen cost prediction mismatch")),
    )

    entry, status = explanations.verified_explanation("2001_2021", "P1", "2023-05-31")
    assert entry is None
    assert status["available"] is False
    assert "mismatch" in status["reason"]
