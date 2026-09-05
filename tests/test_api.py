import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.ml.monthly_lifecycle import SNAPSHOTS, SNAPSHOTS_GZ
from backend.app.services import validation_service

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_temporal_forecast_contract_and_actual_factors():
    response = client.get("/api/projects/701263/forecast")
    assert response.status_code == 200
    data = response.json()
    assert {"project_name", "current_status", "predicted_cost_overrun_percentage", "predicted_delay_days", "risk_score", "risk_level", "explanation"}.issubset(data)
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0 <= data["risk_probability_percentage"] <= 100
    assert data["predicted_delay_days"] >= 0
    assert data["cost_factors"] or data["delay_factors"]
    assert all("feature" in factor and "impact" in factor for factor in data["explanation"])
    assert data["current_progress"] == data["current_status"]["physical_progress_percentage"]
    assert data["predicted_delay_months"] >= 0
    assert data["shap_explanation"] == data["explanation"]


def test_model_performance_and_history_remain_available():
    metrics = client.get("/api/models/metrics")
    assert metrics.status_code == 200
    assert "cost_model" in metrics.json()
    history = client.get("/api/history/705728")
    assert history.status_code == 200
    assert history.json()["snapshots"]


def test_portfolio_uses_temporal_models():
    response = client.get("/api/portfolio/risk?limit=3")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert all(item["model_scope"] == "temporal cost and delay forecasting" for item in items)


def test_historical_portfolio_risk_excludes_project_detail_payloads():
    response = client.get("/api/portfolio/risk?limit=3&window=2001_2021")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert all("cost_factors" not in item and "cost_explanation_summary" not in item for item in items)


def test_prediction_validation_artifacts_are_served():
    report = client.get("/api/models/validation")
    assert report.status_code == 200
    payload = report.json()
    assert payload["cost_model"]["MAE"] >= 0
    risk = payload.get("risk_model", payload.get("risk_classification"))
    assert {"accuracy", "precision", "recall", "f1"}.issubset(risk)
    rows = client.get("/api/models/prediction-validation?limit=5")
    assert rows.status_code == 200
    first = rows.json()["items"][0]
    assert {"project_id", "project_name", "predicted_cost_overrun", "actual_cost_overrun", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error"}.issubset(first)
    rolling = client.get("/api/models/rolling-validation")
    assert rolling.status_code == 200
    assert rolling.json()["fold_count"] == len(rolling.json()["folds"])
    assert {"cost_MAE", "delay_MAE_days", "risk_f1"}.issubset(rolling.json()["folds"][0])


def test_explicit_lifecycle_validation_uses_selected_lifecycle_artifacts(tmp_path, monkeypatch):
    model_root = tmp_path / "monthly_lifecycle" / "2015_2021"
    model_root.mkdir(parents=True)
    lifecycle_metrics = {
        "cost": {"MAE": 41.25, "RMSE": 62.5, "R2": 0.31, "MAPE": 17.2},
        "delay": {"MAE": 510.5, "RMSE": 770.2, "R2": 0.42, "MAPE": 54.1},
        "risk": {"accuracy": 0.61, "macro_precision": 0.49, "macro_recall": 0.52, "macro_f1": 0.47},
    }
    artifact = {
        "window": "2015_2021",
        "metadata": {
            "model_version": "monthly-2015-2021",
            "training_period": [2015, 2021],
            "testing_period": [2022, 2025],
            "unique_training_projects": 11,
            "training_snapshots": 44,
            "unique_test_projects": 2,
            "test_snapshots": 3,
            "features_used": ["approved_cost_cr", "schedule_slippage_days"],
            "feature_availability": {"data_quality_score": 96.4, "removed_invalid_feature_count": 3},
        },
        "lifecycle": {"metrics": lifecycle_metrics},
    }
    (model_root / "evaluation_results.json").write_text(json.dumps(artifact))
    (model_root / "prediction_validation.csv").write_text(
        "canonical_project_id,project_name,completion_year,lifecycle_stage,actual_cost_overrun_percentage,actual_delay_days,predicted_cost_overrun,predicted_delay_days,predicted_risk,actual_risk,cost_error,delay_error\n"
        "LIFE-001,Lifecycle holdout,2023,late,18.0,120.0,15.0,100.0,HIGH,CRITICAL,-3.0,-20.0\n"
    )
    (tmp_path / "validation_report.json").write_text(json.dumps({"cost_model": {"MAE": 2.332}}))
    monkeypatch.setattr(validation_service, "MODELS_DIR", tmp_path)

    report_response = client.get("/api/models/validation?model=2015_2021")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["model_family"] == "monthly_lifecycle"
    assert report["cost_model"] == lifecycle_metrics["cost"]
    assert report["delay_model"] == lifecycle_metrics["delay"]
    assert report["risk_model"] == lifecycle_metrics["risk"]
    assert report["cost_model"]["MAE"] != 2.332
    assert report["metadata"]["training_start"] == 2015
    assert report["metadata"]["training_end"] == 2021
    assert report["metadata"]["evaluated_test_start"] == 2022
    assert report["metadata"]["evaluated_test_end"] == 2025
    assert report["metadata"]["feature_count"] == len(artifact["metadata"]["features_used"])
    assert report["metadata"]["feature_quality"]["data_quality_score"] > 0

    rows_response = client.get("/api/models/prediction-validation?model=2015_2021&limit=5")
    assert rows_response.status_code == 200
    payload = rows_response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["project_id"] == "LIFE-001"
    assert payload["items"][0]["actual_cost_overrun"] == 18.0
    assert payload["items"][0]["completion_year"] == 2023
    assert {"predicted_cost_overrun", "actual_cost_overrun", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error"}.issubset(payload["items"][0])


def test_explicit_missing_model_does_not_fall_back_to_legacy_report():
    response = client.get("/api/models/validation?model=this_model_does_not_exist")
    assert response.status_code == 409
    assert "this_model_does_not_exist" in response.json()["detail"]

    rows = client.get("/api/models/prediction-validation?model=this_model_does_not_exist")
    assert rows.status_code == 409


def test_lifecycle_rolling_validation_is_honest_when_not_generated(tmp_path, monkeypatch):
    model_root = tmp_path / "monthly_lifecycle" / "2015_2021"
    model_root.mkdir(parents=True)
    (model_root / "evaluation_results.json").write_text(json.dumps({"metadata": {}, "lifecycle": {"metrics": {}}}))
    monkeypatch.setattr(validation_service, "MODELS_DIR", tmp_path)

    response = client.get("/api/models/rolling-validation?model=2015_2021")
    assert response.status_code == 200
    assert response.json() == {"model_version": "2015_2021", "folds": [], "fold_count": 0, "status": "not_generated"}


def test_real_paimana_model_simulation_contract():
    versions = client.get("/api/model-simulations")
    assert versions.status_code == 200
    payload = versions.json()
    assert {"2001_2015", "2015_2021"}.issubset({item["key"] for item in payload["items"]})
    assert payload["data_years"]
    assert "lifecycle_data_available" in payload
    simulation = client.post("/api/model-simulations/2001_2015/run")
    assert simulation.status_code == 200
    payload = simulation.json()
    assert payload["metrics"]["metadata"]["data_source"].startswith("Official PAIMANA")
    first = payload["items"][0]
    assert {"predicted_cost_overrun", "actual_cost_overrun", "predicted_delay_days", "actual_delay_days", "shap_explanation"}.issubset(first)


def test_lifecycle_catalog_does_not_label_legacy_years_as_lifecycle(monkeypatch):
    import backend.app.routes.simulations as simulations_route

    def unavailable():
        raise FileNotFoundError("Official processed monthly PAIMANA dataset is unavailable")

    monkeypatch.setattr(simulations_route, "available_data_years", unavailable)
    response = client.get("/api/model-simulations")
    assert response.status_code == 200
    payload = response.json()
    assert payload["lifecycle_data_available"] is False
    assert payload["data_years"] == []
    assert "lifecycle_data_unavailable_reason" in payload


def test_judge_controlled_backtest_hides_actual_until_reveal():
    frozen_validation = Path(__file__).resolve().parents[1] / "models" / "monthly_lifecycle" / "2001_2021" / "prediction_validation.csv"
    if not frozen_validation.exists():
        pytest.skip("requires the local frozen 2001_2021 lifecycle validation artifact")

    trained = client.post("/api/model-simulations/custom/train", json={"start_year": 2001, "end_year": 2015})
    assert trained.status_code == 200, trained.text
    training = trained.json()
    assert training["model_family"] == "monthly_lifecycle"
    assert training["training_samples"] > 0
    assert training["feature_count"] >= 5
    assert training["actual_outcomes_sent_to_browser"] is False
    assert training["eligible_test_years"]

    test_year = training["eligible_test_years"][0]["year"]
    assert test_year > training["training_end"]
    projects = client.get(f"/api/model-simulations/custom/{training['session_id']}/projects?year={test_year}")
    assert projects.status_code == 200
    project_payload = projects.json()
    assert project_payload["actual_outcomes_sent_to_browser"] is False
    selected = project_payload["items"][0]
    assert "actual_cost_overrun" not in selected
    assert "actual_delay_days" not in selected
    assert "completion_date" not in selected

    selection = {"record_index": selected["record_index"]}
    prediction = client.post(f"/api/model-simulations/custom/{training['session_id']}/predict", json=selection)
    assert prediction.status_code == 200
    predicted = prediction.json()
    assert predicted["audit"]["model_family"] == "monthly_lifecycle"
    assert predicted["audit"]["project_excluded_from_training"] is True
    assert predicted["audit"]["actual_outcomes_sent_to_browser"] is False
    assert "actual_cost_overrun" not in predicted
    assert "actual_delay_days" not in predicted
    assert predicted["predicted_risk"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0 <= predicted["risk_probability_percentage"] <= 100

    reveal = client.post(f"/api/model-simulations/custom/{training['session_id']}/reveal", json=selection)
    assert reveal.status_code == 200
    actual = reveal.json()
    assert {"actual_cost_overrun", "actual_delay_days", "cost_error_absolute_pp", "delay_error_absolute_days", "source_url"}.issubset(actual)
    assert actual["cost_error_absolute_pp"] >= 0
    assert actual["delay_error_absolute_days"] >= 0
