from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.lifecycle_run_service import lifecycle_runs


ROOT = Path(__file__).resolve().parents[1]


def test_model_simulation_frontend_does_not_convert_missing_numbers_to_zero():
    source = (ROOT / "frontend" / "src" / "pages" / "ModelSimulationPage.js").read_text()
    assert "Number(value || 0)" not in source
    assert "return parsed === null ? 'N/A'" in source
    assert "run_id" in source
    assert "dataset_fingerprint" in source


def test_registry_uses_canonical_evaluation_metadata_over_poisoned_manifest(tmp_path):
    models_root = tmp_path / "models"
    target = models_root / "monthly_lifecycle" / "2001_2015"
    target.mkdir(parents=True)

    evaluation = {
        "metadata": {
            "model_version": "monthly-2001-2015",
            "training_period": [2001, 2015],
            "testing_period": [2016, 2025],
            "features_used": ["a", "b", "c"],
            "created_at": "2026-08-23T22:15:57+00:00",
            "run_id": "real-run",
            "dataset_fingerprint": "sha256:real",
        },
        "lifecycle": {
            "metrics": {
                "cost": {"MAE": 40.265},
                "delay": {"MAE": 535.48},
                "risk": {"macro_f1": 0.4408},
            }
        },
    }
    (target / "evaluation_results.json").write_text(json.dumps(evaluation))
    (target / "metadata.json").write_text(json.dumps(evaluation["metadata"]))
    (target / "run_manifest.json").write_text(json.dumps({
        "status": "complete",
        "created_at": "2099-01-01T00:00:00+00:00",
        "run_id": "poisoned-run",
        "dataset_fingerprint": "sha256:poisoned",
        "metrics": {"cost_mae": 30.0, "delay_mae": 500.0, "risk_macro_f1": 0.45},
    }))
    (target / "feature_quality_report.json").write_text("{}")
    (target / "prediction_validation.csv").write_text("canonical_project_id,predicted_cost_overrun\nX,1\n")
    for name in ("cost_model.pkl", "delay_model.pkl", "risk_model.pkl"):
        (target / name).write_bytes(b"test")

    item = lifecycle_runs(models_root)["items"][0]
    assert item["cost_mae"] == 40.265
    assert item["delay_mae"] == 535.48
    assert item["risk_macro_f1"] == 0.4408
    assert item["created_at"] == "2026-08-23T22:15:57+00:00"
    assert item["run_id"] == "real-run"
    assert item["dataset_fingerprint"] == "sha256:real"
