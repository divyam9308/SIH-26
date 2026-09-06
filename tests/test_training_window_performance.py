import json

import pytest

from backend.app.services import training_window_performance_service


def _write_evaluation(root, start, end):
    path = root / "monthly_lifecycle" / f"{start}_{end}"
    path.mkdir(parents=True)
    run_id = f"run-{end}"
    fingerprint = "dataset-fingerprint"
    (path / "run_manifest.json").write_text(json.dumps({"status": "complete", "run_id": run_id, "dataset_fingerprint": fingerprint}))
    (path / "evaluation_results.json").write_text(json.dumps({
        "metadata": {"run_id": run_id, "dataset_fingerprint": fingerprint, "testing_period": [end + 1, 2025]},
        "lifecycle": {"metrics": {"cost": {"MAE": end / 10, "R2": 0.4, "unique_projects": end}, "delay": {"MAE": end, "R2": 0.8}}},
    }))


def test_training_window_performance_uses_verified_canonical_evaluations(tmp_path, monkeypatch):
    for end in (2020, 2021, 2022):
        _write_evaluation(tmp_path, 2001, end)
    monkeypatch.setattr(training_window_performance_service, "MODELS_DIR", tmp_path)

    payload = training_window_performance_service.training_window_performance()

    assert [item["end_year"] for item in payload["windows"]] == [2020, 2021, 2022]
    assert all(item["source"] == "verified_canonical_monthly_lifecycle" for item in payload["windows"])
    assert "cohorts differ" in payload["evaluation_period"]


def test_training_window_performance_rejects_unverified_artifact(tmp_path, monkeypatch):
    _write_evaluation(tmp_path, 2001, 2020)
    _write_evaluation(tmp_path, 2001, 2021)
    _write_evaluation(tmp_path, 2001, 2022)
    manifest = tmp_path / "monthly_lifecycle" / "2001_2021" / "run_manifest.json"
    manifest.write_text(json.dumps({"status": "complete", "run_id": "wrong", "dataset_fingerprint": "dataset-fingerprint"}))
    monkeypatch.setattr(training_window_performance_service, "MODELS_DIR", tmp_path)

    with pytest.raises(ValueError, match="invalid run provenance"):
        training_window_performance_service.training_window_performance()
