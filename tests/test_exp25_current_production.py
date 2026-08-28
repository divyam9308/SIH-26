import pandas as pd

from backend.app.ml.experiments.exp25_current_production import (
    EXPERIMENT_ID,
    EXPERIMENT_SEQUENCE,
    MILESTONE_FEATURES,
    add_milestone_features,
    add_project_context_features,
)


def test_exp25_current_identity_and_sequence():
    assert EXPERIMENT_ID == "exp_25_current"
    assert EXPERIMENT_SEQUENCE > 34


def test_project_name_is_parsed_not_retained_as_model_feature():
    frame = pd.DataFrame([{
        "project_name": "Greenfield 6 Lane Highway Phase II Package 3 120 km",
        "sector": "Roads",
        "state": "Gujarat",
        "financial_progress": 40.0,
        "physical_progress": 35.0,
    }])
    out = add_project_context_features(frame)
    assert out.loc[0, "exp25_project_type"] == "road"
    assert out.loc[0, "exp25_has_phase"] == 1.0
    assert out.loc[0, "exp25_phase_number"] == 2.0
    assert out.loc[0, "exp25_package_number"] == 3.0
    assert out.loc[0, "exp25_lane_count"] == 6.0
    assert out.loc[0, "exp25_length_km"] == 120.0
    assert out.loc[0, "exp25_financial_physical_gap"] == 5.0


def test_future_milestone_row_does_not_change_earlier_features():
    history = pd.DataFrame([
        {"canonical_project_id": "p1", "snapshot_date": "2020-01-01", "milestone_status": "1/4"},
        {"canonical_project_id": "p1", "snapshot_date": "2020-02-01", "milestone_status": "2/4"},
        {"canonical_project_id": "p1", "snapshot_date": "2020-03-01", "milestone_status": "4/4"},
    ])
    before = add_milestone_features(history.iloc[:2].copy())
    after = add_milestone_features(history.copy()).iloc[:2].copy()
    for feature in MILESTONE_FEATURES:
        pd.testing.assert_series_equal(
            before[feature].reset_index(drop=True),
            after[feature].reset_index(drop=True),
            check_names=False,
        )
