from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.app.services import lifecycle_retraining_service as retraining
from backend.app.services import lifecycle_simulation_service as simulation


def _comparison_result():
    audit = {"data_quality_score": 97.5, "removed_invalid_feature_count": 2, "removed_features": ["x", "y"], "as_of_evidence_coverage": 100.0}
    provenance = {
        "run_id": "unit-test-run",
        "dataset_fingerprint": "dataset-test-sha",
        "training_fingerprint": "train-test-sha",
        "test_fingerprint": "holdout-test-sha",
        "feature_schema_fingerprint": "schema-test-sha",
        "source_commit": "test-commit",
    }
    metadata = {
        "model_version": "monthly-2001-2015",
        "run_id": provenance["run_id"],
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "training_period": [2001, 2015],
        "testing_period": [2016, 2024],
        "training_snapshots": 100,
        "unique_training_projects": 20,
        "test_snapshots": 30,
        "unique_test_projects": 6,
        "features_used": ["approved_cost_cr", "expenditure_ratio", "sector"],
        "cost_features_used": ["approved_cost_cr", "expenditure_ratio", "sector", "exp12_cost_growth_pct_12m"],
        "delay_features_used": ["approved_cost_cr", "expenditure_ratio", "sector"],
        "risk_features_used": ["approved_cost_cr", "expenditure_ratio", "sector"],
        "production_cost_baseline": "exp12_trajectory_v3_cost_only",
        "promoted_from_experiment": "exp_12",
        "feature_availability": audit,
        "selected_algorithms": {"cost": "xgboost", "delay": "lightgbm"},
        "leakage_policy": "future holdout excluded",
        "snapshot_weighting_policy": "per-project weights",
        "balanced_stage_summary": {"cost_mae": 31.0, "delay_mae": 510.0, "risk_macro_f1": 0.42},
        "provenance": provenance,
    }
    return {
        "metadata": metadata,
        "baseline": {"features": ["approved_cost_cr"], "metrics": {"cost": {"MAE": 50.0}, "delay": {"MAE": 800.0}, "risk": {"macro_f1": 0.2}}},
        "lifecycle": {
            "metrics": {"cost": {"MAE": 30.0}, "delay": {"MAE": 500.0}, "risk": {"macro_f1": 0.45}},
            "lifecycle_stages": {"early": {"available": True}},
            "stage_distribution": {"early": {"rows": 10, "unique_projects": 5}},
            "balanced_stage_summary": metadata["balanced_stage_summary"],
        },
    }


def _write_fake_artifacts(artifact_root, start, end):
    target = artifact_root / f"{start}_{end}"
    target.mkdir(parents=True, exist_ok=True)
    for name in retraining._REQUIRED_ARTIFACTS:
        path = target / name
        if name == "metadata.json":
            path.write_text(json.dumps(_comparison_result()["metadata"]))
        elif name == "evaluation_results.json":
            path.write_text(json.dumps(_comparison_result()))
        else:
            path.write_bytes(b"unit-test-artifact")


def test_year_range_retrain_calls_promoted_production_trainer(tmp_path, monkeypatch):
    data = pd.DataFrame({"completion_year": [2001, 2015, 2020, 2024]})
    identity = pd.DataFrame()
    isolated_root = tmp_path / "monthly_lifecycle"
    monkeypatch.setattr(retraining, "MODEL_ROOT", isolated_root)
    monkeypatch.setattr(retraining, "_training_data", lambda: (data, identity, 2001, 2024))
    called = {}

    def fake_train_window(
        start, end, test_end, data=None, identity=None, artifact_root=None,
        verify_frozen_reference=True,
    ):
        called.update({
            "start": start,
            "end": end,
            "test_end": test_end,
            "data": data,
            "identity": identity,
            "artifact_root": artifact_root,
            "verify_frozen_reference": verify_frozen_reference,
        })
        _write_fake_artifacts(artifact_root, start, end)
        return _comparison_result()

    monkeypatch.setattr(retraining, "train_window_with_promoted_cost_and_delay", fake_train_window)
    # This unit test verifies staging/call routing with intentionally skeletal
    # artifacts; report-content validation is covered by the canonical bundle
    # integration tests and must not require a real retrain here.
    monkeypatch.setattr(retraining, "_write_evaluation_reports", lambda *_args: {})
    result = retraining.retrain_lifecycle(2001, 2015)

    assert called["start"] == 2001
    assert called["end"] == 2015
    assert called["test_end"] == 2024
    assert called["data"] is data
    assert called["verify_frozen_reference"] is False
    assert str(called["artifact_root"]).startswith(str(isolated_root / ".staging"))
    assert result["model_family"] == "monthly_lifecycle"
    assert result["production_cost_baseline"] == "exp12_trajectory_v3_cost_only"
    assert result["promoted_from_experiment"] == "exp_12"
    assert result["cost_features_used"][-1] == "exp12_cost_growth_pct_12m"
    assert result["delay_features_used"] == ["approved_cost_cr", "expenditure_ratio", "sector"]
    assert result["selected_algorithms"] == {"cost": "xgboost", "delay": "lightgbm", "risk": "random_forest"}
    assert result["metrics"]["cost_model"]["MAE"] == 30.0
    assert result["baseline_comparison"]["cost_mae"] == 50.0
    assert result["future_holdout_start"] == 2016
    assert result["run_id"] == "unit-test-run"
    assert (isolated_root / "2001_2015" / "run_manifest.json").exists()
    assert not (isolated_root / "2001_2015" / ".training").exists()


def test_model_simulation_and_models_route_share_current_production_entrypoint():
    from backend.app.routes import models

    assert simulation.retrain_lifecycle is retraining.retrain_lifecycle
    assert models.retrain_lifecycle is retraining.retrain_lifecycle
    assert retraining.train_window_with_promoted_cost_and_delay.__module__.endswith(
        "production_exp105_exp113_baseline"
    )


def test_failed_retrain_always_removes_training_marker(tmp_path, monkeypatch):
    data = pd.DataFrame({"completion_year": [2001, 2015, 2020, 2024]})
    identity = pd.DataFrame()
    isolated_root = tmp_path / "monthly_lifecycle"
    monkeypatch.setattr(retraining, "MODEL_ROOT", isolated_root)
    monkeypatch.setattr(retraining, "_training_data", lambda: (data, identity, 2001, 2024))

    def fail(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(retraining, "train_window_with_promoted_cost_and_delay", fail)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        retraining.retrain_lifecycle(2001, 2015)
    assert not (isolated_root / "2001_2015" / ".training").exists()
    assert not (isolated_root / ".staging").exists() or not any((isolated_root / ".staging").iterdir())


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
            "run_id": "judge-run",
            "dataset_fingerprint": "sha256:judge-dataset",
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
    seen = {}

    def fake_bundle(start, end, expected_run_id=None):
        seen["expected_run_id"] = expected_run_id
        return bundle

    monkeypatch.setattr(simulation, "_artifact_bundle", fake_bundle)
    monkeypatch.setattr(simulation, "_shap_factors_for_model", lambda model, row, names: [{"feature": "revised_cost_cr", "impact": 1.0, "direction": "increases"}])

    session = simulation.train_custom(2001, 2015, "judge-run")
    assert seen["expected_run_id"] == "judge-run"
    assert session["run_id"] == "judge-run"
    assert session["dataset_fingerprint"] == "sha256:judge-dataset"
    assert session["model_family"] == "monthly_lifecycle"
    assert session["feature_count"] == len(features)
    assert session["eligible_test_years"] == [{"year": 2019, "projects": 1}]

    projects = simulation.custom_projects(session["session_id"], 2019)
    assert projects["run_id"] == "judge-run"
    assert projects["dataset_fingerprint"] == "sha256:judge-dataset"
    assert len(projects["items"]) == 1
    assert projects["items"][0]["snapshot_date"] == "2019-09-30"

    prediction = simulation.predict_custom(session["session_id"], projects["items"][0]["record_index"])
    assert prediction["run_id"] == "judge-run"
    assert prediction["dataset_fingerprint"] == "sha256:judge-dataset"
    assert prediction["predicted_cost_overrun"] == 22.0
    assert prediction["predicted_delay_days"] == 360.0
    assert prediction["predicted_risk"] == "HIGH"
    assert prediction["risk_probability_percentage"] == 80.0
    assert prediction["model_inputs"]["revised_cost_cr"] == 250.0
    assert prediction["model_confidence_percentage"] is None
    assert prediction["audit"]["project_excluded_from_training"] is True
    assert prediction["audit"]["actual_outcomes_sent_to_browser"] is False

    actual = simulation.reveal_custom(session["session_id"], prediction["record_index"])
    assert actual["run_id"] == "judge-run"
    assert actual["dataset_fingerprint"] == "sha256:judge-dataset"
    assert actual["actual_cost_overrun"] == 25.0
    assert actual["cost_error_absolute_pp"] == 3.0
    assert actual["actual_outcomes_sent_to_browser"] is True
