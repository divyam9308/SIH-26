import numpy as np
import pandas as pd
import pytest

import backend.app.ml.experiments.recency_delay_exp74 as exp74


def _oof_frame():
    rows = []
    for year in range(2015, 2022):
        for i in range(180):
            rows.append(
                {
                    "oof_year": year,
                    "canonical_project_id": f"P{i % 30:03d}",
                    "sample_weight": 1.0,
                    "actual_delay_days": 100.0,
                    "production_prediction": 90.0,
                    "residual": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_exp74_contract_and_pre_registered_grid():
    assert exp74.EXPERIMENT_SEQUENCE == 74
    assert exp74.EXPERIMENT_SCOPE == "delay"
    assert exp74.WINDOWS == {
        2019: (2020, 2025),
        2021: (2022, 2025),
        2022: (2023, 2025),
        2023: (2024, 2025),
    }
    names = [p["name"] for p in exp74.POLICIES]
    assert names == [
        "full_history_control",
        "rolling_15y",
        "rolling_10y",
        "rolling_7y",
        "rolling_5y",
        "decay_hl_3y",
        "decay_hl_5y",
        "decay_hl_8y",
        "decay_hl_12y",
    ]
    assert exp74.BLEND_GRID == (0.0, 0.25, 0.50, 0.75, 1.0)


def test_window_contract_rejects_unregistered_cutoff():
    assert exp74.window_contract(2021) == (2022, 2025)
    with pytest.raises(ValueError):
        exp74.window_contract(2020)


def test_rolling_policy_is_past_only_and_does_not_mutate_source():
    source = _oof_frame()
    before = source.copy(deep=True)
    policy = {"name": "rolling_5y", "kind": "rolling", "years": 5}
    selected = exp74._policy_frame(source, policy, anchor_end=2021)
    assert selected["oof_year"].min() == 2017
    assert selected["oof_year"].max() == 2021
    pd.testing.assert_frame_equal(source, before)


def test_decay_policy_monotonically_downweights_older_evidence():
    source = pd.DataFrame(
        {
            "oof_year": [2017, 2019, 2021],
            "sample_weight": [1.0, 1.0, 1.0],
            "canonical_project_id": ["A", "B", "C"],
        }
    )
    before = source.copy(deep=True)
    policy = {"name": "decay_hl_3y", "kind": "decay", "half_life": 3.0}
    weighted = exp74._policy_frame(source, policy, anchor_end=2021)
    assert weighted.loc[0, "sample_weight"] < weighted.loc[1, "sample_weight"]
    assert weighted.loc[1, "sample_weight"] < weighted.loc[2, "sample_weight"]
    assert weighted.loc[2, "sample_weight"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(source, before)


def test_aft_selector_relaxes_only_when_verified_limit_is_impossible():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["A", "B", "C"],
            "snapshot_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "planned_completion_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        }
    )
    assert exp74.select_aft_projects_for_exp74(frame, limit=4) == {"A", "B", "C"}


def test_oof_selection_is_conservative_when_all_corrections_are_equal(monkeypatch):
    oof = _oof_frame()

    def zero_correction(fit, score):
        return np.zeros(len(score), dtype=float), {
            "fit_rows": len(fit),
            "fit_projects": fit["canonical_project_id"].nunique(),
            "features": [],
            "training_medians": {},
            "correction_cap_abs_residual_q90": 0.0,
        }

    monkeypatch.setattr(exp74, "_fit_policy_correction", zero_correction)
    selected = exp74.select_recency_policy(oof)
    assert selected["holdout_used_for_selection"] is False
    assert selected["selection_years"] == [2017, 2018, 2019, 2020, 2021]
    assert selected["selected_policy"]["name"] == "full_history_control"
    assert selected["selected_recency_blend_weight"] == 0.0
    assert selected["selected_production_blend_weight"] == 1.0
