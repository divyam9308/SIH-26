from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.milestone_delay_exp25 import (
    ALL_ADDED_FEATURES,
    CONTEXT_FEATURES,
    add_milestone_features,
    add_project_context_features,
    decision,
    enrich_exp25_features,
)


def test_project_name_semantics_are_derived_without_raw_name_feature() -> None:
    frame = pd.DataFrame({
        "project_name": [
            "Greenfield 700 MW Hydro Electric Project Phase II Package 3 - 12.5 km",
            "Six Lane Highway Corridor Extension",
        ],
        "sector": ["Power", "Road"],
        "state": ["Arunachal Pradesh", "Delhi"],
        "physical_progress": [40, 60],
        "financial_progress": [35, 55],
    })
    out = add_project_context_features(frame)
    assert out.loc[0, "exp25_project_type"] == "hydro_power"
    assert out.loc[0, "exp25_capacity_mw"] == 700
    assert out.loc[0, "exp25_phase_number"] == 2
    assert out.loc[0, "exp25_package_number"] == 3
    assert out.loc[0, "exp25_length_km"] == 12.5
    assert out.loc[0, "exp25_has_greenfield"] == 1
    assert out.loc[1, "exp25_project_type"] == "road"
    assert out.loc[1, "exp25_has_corridor"] == 1
    assert out.loc[1, "exp25_has_extension"] == 1
    assert out.loc[0, "exp25_financial_physical_gap"] == -5
    assert "project_name" not in CONTEXT_FEATURES
    assert "project_name" not in ALL_ADDED_FEATURES


def test_structured_project_context_includes_supported_paimana_fields() -> None:
    required = {
        "sector",
        "ministry",
        "implementing_agency",
        "state",
        "approved_cost_cr",
        "revised_cost_cr",
        "cumulative_expenditure_cr",
        "physical_progress",
        "current_schedule_status",
    }
    assert required.issubset(set(CONTEXT_FEATURES))


def test_milestone_parser_and_past_only_velocity() -> None:
    frame = pd.DataFrame({
        "canonical_project_id": ["P1", "P1", "P1"],
        "snapshot_date": ["2020-01-01", "2020-04-01", "2020-07-01"],
        "milestone_status": ["2/10", "4/10", "4/10"],
    })
    out = add_milestone_features(frame)
    assert out.loc[0, "exp25_milestone_ratio"] == 0.2
    assert out.loc[1, "exp25_milestone_delta"] == 2
    assert out.loc[1, "exp25_milestone_velocity"] > 0
    assert out.loc[2, "exp25_milestone_stagnant"] == 1
    assert out.loc[2, "exp25_months_since_milestone_change"] > 0


def test_supervised_snapshot_uses_full_monthly_history_and_context() -> None:
    supervised = pd.DataFrame({
        "canonical_project_id": ["P1"],
        "snapshot_date": ["2020-04-01"],
        "project_name": ["Metro Corridor Phase 2"],
        "sector": ["Urban Transport"],
        "state": ["Delhi"],
        "physical_progress": [30],
        "financial_progress": [25],
    })
    monthly = pd.DataFrame({
        "canonical_project_id": ["P1", "P1", "P1", "P1"],
        "snapshot_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "milestone_status": ["2/10", "2/10", "3/10", "4/10"],
    })
    out = enrich_exp25_features(supervised, history=monthly).iloc[0]
    assert out.exp25_milestone_delta == 1
    assert out.exp25_milestone_velocity > 0
    assert out.exp25_project_type == "metro"
    assert out.exp25_phase_number == 2
    assert out.exp25_financial_physical_gap == -5


def test_future_row_cannot_change_earlier_milestone_features() -> None:
    base = pd.DataFrame({
        "canonical_project_id": ["P1", "P1", "P1"],
        "snapshot_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "milestone_status": ["1/10", "2/10", "3/10"],
    })
    before = add_milestone_features(base)
    with_future = pd.concat([
        base,
        pd.DataFrame({
            "canonical_project_id": ["P1"],
            "snapshot_date": ["2030-01-01"],
            "milestone_status": ["10/10"],
        }),
    ], ignore_index=True)
    after = add_milestone_features(with_future).iloc[:3]
    cols = [
        "exp25_milestone_ratio",
        "exp25_milestone_velocity",
        "exp25_milestone_delta",
        "exp25_milestone_stagnant",
        "exp25_months_since_milestone_change",
    ]
    pd.testing.assert_frame_equal(
        before[cols].reset_index(drop=True),
        after[cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_combined_promotion_rule() -> None:
    assert decision(0.0, 0.01) == "PROMOTION CANDIDATE"
    assert decision(0.01, 0.0) == "PROMOTION CANDIDATE"
    assert decision(0.01, 0.01) == "PROMOTION CANDIDATE"
    assert decision(0.0, 0.0) == "REGRESSION / DO NOT PROMOTE"
    assert decision(-0.01, 1.0) == "REGRESSION / DO NOT PROMOTE"
    assert decision(1.0, -0.01) == "REGRESSION / DO NOT PROMOTE"
