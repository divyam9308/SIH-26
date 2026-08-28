import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    EXPERIMENT_SCOPE,
    FAMILIES,
    PATH_FEATURES,
    _path_history,
    _weight_grid,
)


def _history():
    return pd.DataFrame({
        "canonical_project_id": ["A", "A", "A", "B", "B"],
        "snapshot_date": ["2020-01-31", "2020-02-29", "2020-03-31", "2020-01-31", "2020-02-29"],
        "revised_cost_cr": [100, 110, 120, 200, 200],
        "cost_escalation_percentage": [0, 10, 20, 0, 0],
        "schedule_slippage_days": [0, 30, 60, 10, 10],
    })


def test_exp34_is_delay_only_and_weight_grid_is_convex():
    assert EXPERIMENT_SCOPE == "delay"
    for row in _weight_grid():
        assert set(row) == set(FAMILIES)
        assert all(value >= 0 for value in row.values())
        assert abs(sum(row.values()) - 1.0) < 1e-12


def test_path_features_are_causal_under_future_append():
    base = _history()
    before = _path_history(base)
    future = pd.DataFrame({
        "canonical_project_id": ["A"],
        "snapshot_date": ["2030-01-31"],
        "revised_cost_cr": [99999],
        "cost_escalation_percentage": [9999],
        "schedule_slippage_days": [99999],
    })
    after = _path_history(pd.concat([base, future], ignore_index=True))
    key = ["canonical_project_id", "snapshot_date"]
    merged = before.merge(after, on=key, suffixes=("_before", "_after"))
    for feature in PATH_FEATURES:
        left = merged[f"{feature}_before"]
        right = merged[f"{feature}_after"]
        pd.testing.assert_series_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_names=False)
