from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments.trajectory_exp13_v2 import (
    CHANGE_POINT_FEATURES,
    REGIME_FEATURES,
    apply_regime_encoder,
    engineer_change_points,
    fit_regime_encoder,
    rolling_temporal_folds,
    stage_aware_training_weights,
)


def _history(project: str = "P") -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    revised = [100, 100, 100, 101, 102, 103, 104, 105, 110, 120, 133, 148, 164, 182]
    spend = [5, 10, 16, 22, 29, 36, 43, 50, 56, 61, 66, 70, 73, 75]
    slip = [0, 0, 0, 2, 4, 7, 10, 14, 25, 45, 75, 110, 150, 195]
    for i, stamp in enumerate(dates):
        rows.append(
            {
                "canonical_project_id": project,
                "snapshot_date": stamp,
                "approved_cost_cr": 100.0,
                "revised_cost_cr": float(revised[i]),
                "cumulative_expenditure_cr": float(spend[i]),
                "schedule_slippage_days": float(slip[i]),
                "planned_duration_days": 730.0,
                "expected_progress_percentage": float(min(95, 5 + i * 6)),
                "planned_completion_date": "2022-12-31",
                "revised_completion_date": "2022-12-31" if i < 8 else "2023-08-31",
            }
        )
    return pd.DataFrame(rows)


def test_online_change_points_are_as_of_safe_when_future_report_is_added():
    history = _history()
    before = engineer_change_points(history)
    future = history.iloc[-1].copy()
    future["snapshot_date"] = pd.Timestamp("2021-03-31")
    future["revised_cost_cr"] = 5000.0
    future["cumulative_expenditure_cr"] = 5000.0
    future["schedule_slippage_days"] = 5000.0
    after = engineer_change_points(pd.concat([history, pd.DataFrame([future])], ignore_index=True))

    cutoff = history.snapshot_date.max()
    before_rows = before[pd.to_datetime(before.snapshot_date).le(cutoff)].reset_index(drop=True)
    after_rows = after[pd.to_datetime(after.snapshot_date).le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_rows[CHANGE_POINT_FEATURES], after_rows[CHANGE_POINT_FEATURES])


def test_learned_regime_encoder_produces_soft_probabilities_from_training_fit_only():
    histories = []
    for i in range(6):
        frame = _history(f"P{i}")
        frame["revised_cost_cr"] = frame["revised_cost_cr"] * (1.0 + i * 0.03)
        frame["schedule_slippage_days"] = frame["schedule_slippage_days"] * (1.0 + i * 0.08)
        histories.append(frame)
    engineered = engineer_change_points(pd.concat(histories, ignore_index=True))
    engineered["completion_year"] = 2020 + (np.arange(len(engineered)) % 3)
    engineered["sample_weight"] = 1.0
    engineered["lifecycle_stage"] = "mid"

    fitting = engineered.iloc[:-10].copy()
    validation = engineered.iloc[-10:].copy()
    encoder = fit_regime_encoder(fitting, seed=7)
    transformed = apply_regime_encoder(validation, encoder)

    assert set(REGIME_FEATURES).issubset(transformed.columns)
    probabilities = transformed[[f"exp13v2_regime_probability_{i}" for i in range(4)]].to_numpy(float)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(transformed)), atol=1e-6)
    assert transformed.exp13v2_regime_confidence.between(0, 1).all()
    assert transformed.exp13v2_regime_entropy.ge(0).all()


def test_rolling_temporal_selection_uses_multiple_forward_only_folds():
    rows = []
    for year in range(2005, 2020):
        for project in range(3):
            rows.append(
                {
                    "completion_year": year,
                    "canonical_project_id": f"{year}-{project}",
                }
            )
    frame = pd.DataFrame(rows)
    folds = rolling_temporal_folds(frame, max_folds=3)
    assert len(folds) == 3
    for fit_years, validation_years in folds:
        assert max(fit_years) < min(validation_years)


def test_stage_aware_training_weights_prioritize_early_and_mid_over_very_late():
    frame = pd.DataFrame(
        {
            "sample_weight": [1.0, 1.0, 1.0, 1.0],
            "lifecycle_stage": ["early", "mid", "late", "very_late"],
        }
    )
    weighted = stage_aware_training_weights(frame)
    values = weighted.sample_weight.to_numpy(float)
    assert values[0] > values[2] > values[3]
    assert values[1] > values[2]
    assert np.isclose(values.mean(), 1.0)
