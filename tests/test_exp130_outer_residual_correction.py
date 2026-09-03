from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp130_outer_residual_correction import (
    FORBIDDEN_MODEL_FEATURES,
    RESIDUAL_FEATURES,
    TEST_END,
    TEST_START,
    TRAINING_END,
    TRAINING_START,
    add_residual_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_date": ["2021-01-01", "2021-06-01"],
            "planned_completion_date": ["2022-01-01", "2022-01-01"],
            "revised_completion_date": ["2022-03-01", "2022-06-01"],
            "planned_duration_days": [1000, 1000],
            "elapsed_duration_days": [300, 800],
            "duration_ratio": [0.30, 0.80],
            "physical_progress": [10.0, 90.0],
            "expenditure_ratio": [0.15, 0.85],
            "approved_cost_cr": [500.0, 500.0],
            "revised_cost_cr": [550.0, 650.0],
            "cost_escalation_percentage": [10.0, 30.0],
            "progress_deviation": [-20.0, 10.0],
            "schedule_slippage_days": [60.0, 150.0],
            "sector": ["Railways", "Railways"],
            "implementing_agency": ["Agency A", "Agency A"],
            "identity_confidence": [1.0, 1.0],
            "exp58_group_support": [10.0, 20.0],
            "cost_growth_velocity_3m": [1.0, 2.0],
            "cost_growth_velocity_6m": [1.5, 2.5],
            "cost_acceleration": [0.1, 0.2],
            "progress_velocity_3m": [2.0, 3.0],
            "progress_velocity_6m": [1.0, 2.0],
            "progress_acceleration": [0.2, 0.3],
            "actual_cost_overrun_percentage": [999.0, -999.0],
            "actual_delay_days": [9999.0, 1.0],
        }
    )


def test_exp130_window_is_exactly_2001_2021_to_2022_2025():
    assert (TRAINING_START, TRAINING_END, TEST_START, TEST_END) == (2001, 2021, 2022, 2025)


def test_residual_model_feature_contract_contains_no_target_or_error():
    assert FORBIDDEN_MODEL_FEATURES.isdisjoint(RESIDUAL_FEATURES)


def test_feature_engineering_is_target_independent():
    frame = _frame()
    prediction = np.array([100.0, 100.0])
    left = add_residual_features(frame, prediction)[RESIDUAL_FEATURES].copy()

    changed = frame.copy()
    changed["actual_cost_overrun_percentage"] = [-1e9, 1e9]
    changed["actual_delay_days"] = [1e9, -1e9]
    right = add_residual_features(changed, prediction)[RESIDUAL_FEATURES].copy()

    pd.testing.assert_frame_equal(left, right)


def test_prediction_lifecycle_interaction_changes_by_project_stage():
    features = add_residual_features(_frame(), np.array([20.0, 20.0]))
    assert features.loc[0, "production_x_lifecycle_stage"] != features.loc[1, "production_x_lifecycle_stage"]
    assert features.loc[0, "lifecycle_stage"] == "EARLY"
    assert features.loc[1, "lifecycle_stage"] == "LATE"


def test_confidence_proxy_is_bounded():
    confidence = add_residual_features(_frame(), np.array([1.0, 1.0]))["prediction_confidence_proxy"]
    assert confidence.between(0.0, 1.0).all()
