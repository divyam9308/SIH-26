import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import portfolio_service
from backend.app.services import range_portfolio_service


client = TestClient(app)


def test_paginated_project_contract_matches_five_direct_forecasts():
    response = client.get("/api/projects?page=1&page_size=5&sort=score&direction=desc")
    assert response.status_code == 200
    payload = response.json()
    assert {"items", "total", "page", "page_size", "pages", "sectors", "ministries", "model_version", "dataset_snapshot"}.issubset(payload)
    assert len(payload["items"]) == 5
    for item in payload["items"]:
        forecast = client.get(f"/api/projects/{item['project_code']}/forecast")
        assert forecast.status_code == 200
        authoritative = forecast.json()
        assert item["predicted_cost_overrun_percentage"] == authoritative["predicted_cost_overrun_percentage"]
        assert item["predicted_delay_days"] == authoritative["predicted_delay_days"]
        assert item["risk_level"] == authoritative["risk_level"]
        assert item["risk_score"] == authoritative["risk_score"]
        assert item["model_version"] == authoritative["model_version"]
        assert "cost_factors" not in item
        assert "shap_explanation" not in item


def test_project_contract_preserves_nulls_and_forecast_provenance():
    project = client.get("/api/projects/602098")
    assert project.status_code == 200
    assert project.json()["physical_progress_pct"] is None
    assert project.json()["implementing_agency"] is None
    forecast = client.get("/api/projects/701263/forecast")
    assert forecast.status_code == 200
    payload = forecast.json()
    assert {"model_version", "dataset_snapshot_date", "inference_timestamp", "model_scope", "confidence_calibration_status"}.issubset(payload)
    assert payload["predicted_final_cost_cr"] == round(43129 + payload["predicted_cost_overrun_amount_cr"], 2)


def test_project_filters_and_empty_result_contract():
    filtered = client.get("/api/projects?page=1&page_size=10&search=Rajasthan%20Refinery")
    assert filtered.status_code == 200
    assert [item["project_code"] for item in filtered.json()["items"]] == ["701263"]
    empty = client.get("/api/projects?page=1&page_size=10&search=no-such-real-project-xyz")
    assert empty.status_code == 200
    assert empty.json()["items"] == []
    assert empty.json()["total"] == 0


def test_portfolio_cache_key_includes_model_and_dataset_signatures(monkeypatch):
    frame = pd.DataFrame([{
        "project_code": "X1", "project_name": "Example", "sector": "Railways", "ministry": "Railways",
        "snapshot_date": pd.Timestamp("2026-05-31"), "original_cost_cr": 100.0, "revised_cost_cr": None,
        "expenditure_cr": 20.0, "physical_progress_pct": None, "schedule_extension_days": None,
        "cost_escalation_pct": None, "financial_progress_pct": 20.0, "revised_end_date": None,
    }])
    calls = {"forecast": 0}
    def fake_forecast(_code, include_explanations=True):
        calls["forecast"] += 1
        assert include_explanations is False
        return {
            "predicted_delay_days": 30.0, "predicted_cost_overrun_percentage": 10.0,
            "predicted_cost_overrun_amount_cr": 10.0, "predicted_final_cost_cr": 110.0,
            "predicted_delay_months": 1.0, "predicted_completion_date": "2027-01-01",
            "risk_score": 70.0, "risk_probability_percentage": 70.0, "risk_level": "HIGH",
            "model_scope": "scope", "model_confidence_percentage": None,
            "confidence_calibration_status": "unavailable", "best_models": {"cost": "cost", "delay": "delay"},
        }
    monkeypatch.setattr(portfolio_service, "projects_df", lambda: frame)
    monkeypatch.setattr(portfolio_service, "project_forecast", fake_forecast)
    portfolio_service.invalidate_portfolio_cache()
    portfolio_service._portfolio_payload_cached("v1", "model-a", "data-a")
    portfolio_service._portfolio_payload_cached("v1", "model-a", "data-a")
    assert calls["forecast"] == 1
    portfolio_service._portfolio_payload_cached("v2", "model-b", "data-a")
    portfolio_service._portfolio_payload_cached("v2", "model-b", "data-b")
    assert calls["forecast"] == 3
    portfolio_service.invalidate_portfolio_cache()


def test_saved_historical_windows_expose_precomputed_project_views():
    expected = {"2001_2017": 1233, "2001_2021": 728, "2001_2022": 489}
    for window, count in expected.items():
        summary = client.get("/api/portfolio/summary", params={"window": window})
        projects = client.get("/api/projects", params={"window": window, "page_size": 1})
        assert summary.status_code == 200
        assert projects.status_code == 200
        assert summary.json()["projects"] == count
        assert projects.json()["total"] == count


def test_saved_historical_views_are_repository_relative_not_cwd(monkeypatch, tmp_path):
    expected = Path(__file__).resolve().parents[1] / "data" / "processed" / "portfolio_windows"
    assert range_portfolio_service.SAVED_WINDOW_ROOT == expected
    monkeypatch.chdir(tmp_path)
    range_portfolio_service.invalidate_range_cache()
    response = client.get("/api/projects", params={"window": "2001_2017", "page_size": 1})
    assert response.status_code == 200
    assert response.json()["total"] == 1233
    range_portfolio_service.invalidate_range_cache()


def test_missing_historical_artifacts_return_controlled_response(monkeypatch, tmp_path):
    monkeypatch.setattr(range_portfolio_service, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(range_portfolio_service, "SAVED_WINDOW_ROOT", tmp_path / "saved")
    range_portfolio_service.invalidate_range_cache()
    response = client.get("/api/projects", params={"window": "2001_2017", "page_size": 1})
    assert response.status_code == 409
    assert "Production evaluation" in response.json()["detail"]
    range_portfolio_service.invalidate_range_cache()


def test_historical_project_rows_include_actuals_and_prediction_errors():
    response = client.get("/api/projects", params={"window": "2001_2021", "page_size": 1})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert {"actual_cost_overrun_percentage", "actual_delay_days", "cost_error_percentage", "delay_error_days"}.issubset(item)
    assert item["risk_score"] >= 90
    assert item["risk_level"] == "CRITICAL"


def test_historical_project_detail_honours_selected_window():
    item = client.get("/api/projects", params={"window": "2001_2021", "page_size": 1}).json()["items"][0]
    code = item["project_code"]
    detail = client.get(f"/api/projects/{code}", params={"window": "2001_2021"})
    forecast = client.get(f"/api/projects/{code}/forecast", params={"window": "2001_2021"})
    assert detail.status_code == 200
    assert forecast.status_code == 200
    assert detail.json()["project_code"] == code
    assert forecast.json()["model_version"] == item["model_version"]


def test_prediction_accuracy_endpoints_resolve_the_requested_lifecycle_window():
    report = client.get("/api/models/validation", params={"model_version": "2001_2021"})
    rows = client.get("/api/models/prediction-validation", params={"model_version": "2001_2021", "limit": 1})
    importance = client.get("/api/models/importance", params={"model_version": "2001_2021"})
    assert report.status_code == rows.status_code == importance.status_code == 200
    assert report.json()["model_version"] == "monthly-2001-2021"
    assert report.json()["metadata"]["training_start"] == 2001
    assert report.json()["metadata"]["training_end"] == 2021
    assert report.json()["metadata"]["test_start"] == 2022
    assert report.json()["metadata"]["test_end"] == 2025
    assert rows.json()["model_version"] == "2001_2021"
    assert importance.json()["model_version"] == "monthly-2001-2021"


def test_prediction_accuracy_legacy_window_exposes_its_own_holdout_artifacts():
    client = TestClient(app)
    report = client.get("/api/models/validation", params={"model_version": "2001_2022"})
    rows = client.get("/api/models/prediction-validation", params={"model_version": "2001_2022", "limit": 1})

    assert report.status_code == 200
    assert rows.status_code == 200
    assert report.json()["model_version"] == "2001_2022"
    assert report.json()["metadata"]["training_end"] == 2022
    assert report.json()["metadata"]["test_start"] == 2023
    assert rows.json()["model_version"] == "2001_2022"
    assert rows.json()["total"] == 42
