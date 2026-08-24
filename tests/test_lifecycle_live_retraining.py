from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.services import lifecycle_retraining_service as retraining
from backend.app.services import lifecycle_simulation_service as simulation


def _comparison_result():
    audit = {"data_quality_score": 97.5, "removed_invalid_feature_count": 2, "removed_features": ["x", "y"]}
    metadata = {
        "model_version": "monthly-2001-2015",
        "training_period": [2001, 2015],
        "testing_period": [2016, 2024],
        "training_snapshots": 100,
        "unique_training_projects": 20,
        "test_snapshots": 30,
        "unique_test_projects": 6,
        "features_used": ["approved_cost_cr", "expenditure_ratio", "sector"],
        "feature_availability": audit,
        "selected_algorithms": {"cost": "xgboost", "delay": "lightgbm"},
        "leakage_policy": "future holdout excluded",
        "snapshot_weighting_policy": "per-project weights",
        "created_at": "2026-08-24T00:00:00+00:00",
    }
    return {
        "metadata": metadata,
        "baseline": {"metrics": {"cost": {"MAE": 50.0}, "delay": {"MAE": 800.0}, "risk": {"macro_f1": 0.2}}},
        "lifecycle": {
            "metrics": {"cost": {"MAE": 30.0}, "delay": {"MAE": 500.0}, "risk": {"macro_f1": 0.45}},
            "lifecycle_stages": {"early": {"available": True}},
        },
    }


def test_year_range_retrain_calls_monthly_lifecycle_trainer(monkeypatch, tmp_path):
    data = pd.DataFrame({"completion_year": [2001, 2015, 2020, 2024]})
    identity = pd.DataFrame()
    isolated_root = tmp_path / "models" / "monthly_lifecycle"
    monkeypatch.setattr(retraining, "MODEL_ROOT", isolated_root)
    monkeypatch.setattr(retraining, "_training_data", lambda: (data, identity, 2001, 2024))
    called = {}

    def fake_train_window(start, end, test_end, data=None, identity=None):
        called.update({"start": start, "end": end, "test_end": test_end, "data": data, "identity": identity})
        return _comparison_result()

    monkeypatch.setattr(retraining, "train_window", fake_train_window)
    result = retraining.retrain_lifecycle(2001, 2015)

    assert called["start"] == 2001
    assert called["end"] == 2015
    assert called["test_end"] == 2024
    assert called["data"] is data
    assert result["model_family"] == "monthly_lifecycle"
    assert result["selected_algorithms"] == {"cost": "xgboost", "delay": "lightgbm", "risk": "random_forest"}
    assert result["metrics"]["cost_model"]["MAE"] == 30.0
    assert result["baseline_comparison"]["cost_mae"] == 50.0
    assert result["future_holdout_start"] == 2016
    assert result["run_id"]
    assert result["dataset_fingerprint"].startswith("sha256:")

    manifest_path = isolated_root / "2001_2015" / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == result["run_id"]
    assert manifest["metrics"]["cost_mae"] == 30.0
    assert manifest["provenance_status"] == "verified_runtime_run"


class _Regressor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.array([self.value] * len(frame))


class _Risk:
    def predict(self, frame):
        return np.array(["HIGH"] * len(frame), dtype=object)

    def predict_proba(self, frame):
        return np.tile(np.array([[0.05, 0.1, 0.8, 0.05]]), (len(frame), 1))


def test_custom_judge_prediction_uses_exact_lifecycle_run_and_future_project(monkeypatch):
    frame = pd.DataFrame([
        {
            "canonical_project_id": "TRAIN",
            "project_id": "N00000001",
            "project_name": "Training Project",
            "completion_year": 2015,
            "completion_date": pd.Timestamp("2015-12-31"),
            "snapshot_date": pd.Timestamp("2015-06-30"),
            "approved_cost_cr": 100.0,
            "revised_cost_cr": 110.0,
            "expenditure_ratio": 0.5,
            "sector": "Power",
            "actual_cost_overrun_percentage": 10.0,
            "actual_delay_days": 100.0,
            "actual_risk": "MEDIUM",
        },
        {
            "canonical_project_id": "TEST",
            "project_id": "N00000002",
            "project_name": "Held-out Project",
            "completion_year": 2019,
            "completion_date": pd.Timestamp("2019-12-31"),
            "snapshot_date": pd.Timestamp("2019-03-31"),
            "approved_cost_cr": 200.0,
            "revised_cost_cr": 240.0,
            "expenditure_ratio": 0.7,
            "sector": "Railways",
            "actual_cost_overrun_percentage": 25.0,
            "actual_delay_days": 420.0,
            "actual_risk": "HIGH",
        },
        {
            "canonical_project_id": "TEST",
            "project_id": "N00000002",
            "project_name": "Held-out Project",
            "completion_year": 2019,
            "completion_date": pd.Timestamp("2019-12-31"),
            "snapshot_date": pd.Timestamp("2019-09-30"),
            "approved_cost_cr": 200.0,
            "revised_cost_cr": 250.0,
            "expenditure_ratio": 0.9,
            "sector": "Railways",
            "actual_cost_overrun_percentage": 25.0,
            "actual_delay_days": 420.0,
            "actual_risk": "HIGH",
        },
    ])
    features = ["approved_cost_cr", "revised_cost_cr", "expenditure_ratio", "sector"]
    bundle = {
        "metadata": {
            "model_version": "monthly-2001-2015",
            "run_id": "run-test-123",
            "dataset_fingerprint": "sha256:testfingerprint",
            "training_snapshots": 1,
            "unique_training_projects": 1,
            "features_used": features,
            "selected_algorithms": {"cost": "xgboost", "delay": "lightgbm"},
        },
        "cost": _Regressor(22.0),
        "delay": _Regressor(360.0),
        "risk": _Risk(),
    }
    simulation._CUSTOM_SESSIONS.clear()
    monkeypatch.setattr(simulation, "_dataset", lambda: frame)
    monkeypatch.setattr(simulation, "_artifact_bundle", lambda start, end, expected_run_id=None: bundle)
    monkeypatch.setattr(simulation, "_shap_factors_for_model", lambda model, row, names: [{"feature": "revised_cost_cr", "impact": 1.0, "direction": "increases"}])

    session = simulation.train_custom(2001, 2015, "run-test-123")
    assert session["model_family"] == "monthly_lifecycle"
    assert session["run_id"] == "run-test-123"
    assert session["dataset_fingerprint"] == "sha256:testfingerprint"
    assert session["feature_count"] == len(features)
    assert session["eligible_test_years"] == [{"year": 2019, "projects": 1}]

    projects = simulation.custom_projects(session["session_id"], 2019)
    assert len(projects["items"]) == 1
    assert projects["run_id"] == "run-test-123"
    assert projects["items"][0]["snapshot_date"] == "2019-09-30"

    prediction = simulation.predict_custom(session["session_id"], projects["items"][0]["record_index"])
    assert prediction["run_id"] == "run-test-123"
    assert prediction["predicted_cost_overrun"] == 22.0
    assert prediction["predicted_delay_days"] == 360.0
    assert prediction["predicted_risk"] == "HIGH"
    assert prediction["risk_probability_percentage"] == 80.0
    assert prediction["model_inputs"]["revised_cost_cr"] == 250.0
    assert prediction["model_confidence_percentage"] is None
    assert prediction["audit"]["project_excluded_from_training"] is True
    assert prediction["audit"]["actual_outcomes_sent_to_browser"] is False

    actual = simulation.reveal_custom(session["session_id"], prediction["record_index"])
    assert actual["run_id"] == "run-test-123"
    assert actual["actual_cost_overrun"] == 25.0
    assert actual["cost_error_absolute_pp"] == 3.0
    assert actual["actual_outcomes_sent_to_browser"] is True
