import pandas as pd
from pathlib import Path

from backend.app.services.operational_driver_service import operational_drivers


def _record(**overrides):
    row = {
        "snapshot_date": "2023-12-31", "planned_completion_date": "2020-01-01",
        "revised_completion_date": "2022-01-01", "actual_completion_date": None,
        "approved_cost_cr": 100.0, "revised_cost_cr": 125.0,
        "physical_progress": 30.0, "financial_progress": 60.0,
        "expected_progress_percentage": 80.0,
    }
    row.update(overrides)
    return row


def test_derives_evidence_backed_progress_cost_and_schedule_signals():
    drivers = operational_drivers(_record(), source="official_snapshot_trajectory")
    kinds = {item["type"] for item in drivers}
    assert {"SCHEDULE_EXTENSION", "COST_REVISION", "EXPENDITURE_PROGRESS_MISMATCH", "PHYSICAL_PROGRESS_LAG", "PLANNED_DEADLINE_CROSSED"} == kinds
    mismatch = next(item for item in drivers if item["type"] == "EXPENDITURE_PROGRESS_MISMATCH")
    assert "30.0 percentage points" in mismatch["evidence"]
    assert mismatch["provenance"] == "derived"


def test_missing_values_do_not_become_zero_or_create_drivers():
    drivers = operational_drivers(_record(
        planned_completion_date=None, revised_completion_date=None, approved_cost_cr=None,
        revised_cost_cr=None, physical_progress=None, financial_progress=None,
        expected_progress_percentage=None,
    ), source="official_snapshot_trajectory")
    assert drivers == []


def test_history_detects_real_date_revisions_and_stagnant_progress():
    history = pd.DataFrame([
        _record(snapshot_date="2022-01-01", revised_completion_date="2023-01-01", physical_progress=35, financial_progress=None, expected_progress_percentage=None, approved_cost_cr=None, revised_cost_cr=None),
        _record(snapshot_date="2022-06-01", revised_completion_date="2024-01-01", physical_progress=35.5, financial_progress=None, expected_progress_percentage=None, approved_cost_cr=None, revised_cost_cr=None),
        _record(snapshot_date="2023-01-01", revised_completion_date="2025-01-01", physical_progress=35.5, financial_progress=None, expected_progress_percentage=None, approved_cost_cr=None, revised_cost_cr=None),
    ])
    drivers = operational_drivers(history.iloc[-1], history, source="official_snapshot_trajectory")
    assert {item["type"] for item in drivers} >= {"REPEATED_COMPLETION_REVISION", "STAGNANT_PROGRESS"}


def test_no_free_text_cause_is_inferred_without_a_supported_dataset_field():
    drivers = operational_drivers(_record(remarks="land acquisition pending"), source="official_snapshot_trajectory")
    assert "LAND_ACQUISITION" not in {item["type"] for item in drivers}


def test_ranking_is_deterministic_and_limited_to_five():
    history = pd.DataFrame([
        _record(snapshot_date="2021-01-01", revised_completion_date="2021-01-01", physical_progress=30),
        _record(snapshot_date="2022-01-01", revised_completion_date="2022-01-01", physical_progress=30),
        _record(snapshot_date="2023-01-01", revised_completion_date="2023-01-01", physical_progress=30),
    ])
    first = operational_drivers(history.iloc[-1], history, source="official_snapshot_trajectory")
    second = operational_drivers(history.iloc[-1], history, source="official_snapshot_trajectory")
    assert first == second
    assert len(first) == 5


def test_project_detail_keeps_model_evidence_and_renders_operational_drivers():
    page = (Path(__file__).resolve().parents[1] / "frontend/src/pages/ProjectDetail.tsx").read_text()
    assert "Model Evidence" in page
    assert "Cost SHAP factors" in page
    assert "Delay SHAP factors" in page
    assert "Risk SHAP factors" in page
    assert "Operational Drivers" in page
    assert "Future Integration — Administrative Cause Intelligence" in page
    assert "Conceptual future capability" in page
    assert "Land acquisition" in page
    assert "No verified operational drivers were available" in page
