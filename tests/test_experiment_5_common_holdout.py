from __future__ import annotations

import pandas as pd
import pytest

from backend.app.ml.experiments.common_holdout_training_window import (
    EXPECTED_FEATURES, calculate_improvements, decide_winner, make_splits,
)
from backend.app.ml.monthly_training import _balanced_stage_summary
from backend.app.ml.provenance import frame_fingerprint


def cohort():
    rows = []
    for project, year, count in [("A", 2019, 2), ("B", 2021, 2), ("C", 2022, 2), ("D", 2025, 1)]:
        for snapshot in range(count):
            rows.append({"canonical_project_id": project, "completion_year": year, "snapshot_date": f"{year}-01-0{snapshot + 1}", "sample_weight": 1 / count})
    return pd.DataFrame(rows)


def test_experiment_windows_and_single_common_holdout():
    a, b, test = make_splits(cohort())
    assert set(a.completion_year) <= set(range(2001, 2020))
    assert set(b.completion_year) <= set(range(2001, 2022))
    assert set(test.completion_year) <= set(range(2022, 2026))
    assert set(a.canonical_project_id).isdisjoint(test.canonical_project_id)
    assert set(b.canonical_project_id).isdisjoint(test.canonical_project_id)
    assert frame_fingerprint(test) == frame_fingerprint(test.copy())
    assert set(test.canonical_project_id) == {"C", "D"}


def test_expected_contract_is_exactly_25_features():
    assert len(EXPECTED_FEATURES) == 25
    assert len(set(EXPECTED_FEATURES)) == 25


def metric_result(cost, delay, risk, early_cost=None, early_delay=None, mid_cost=None, mid_delay=None, balanced_cost=None, balanced_delay=None):
    stage = {}
    for name, c, d in (("early", early_cost, early_delay), ("mid", mid_cost, mid_delay)):
        stage[name] = {"available": c is not None and d is not None, "cost": {"MAE": c} if c is not None else {}, "delay": {"MAE": d} if d is not None else {}}
    return {"lifecycle_metrics": {"cost": {"MAE": cost}, "delay": {"MAE": delay}, "risk": {"macro_f1": risk}}, "balanced_stage_summary": {"cost_mae": balanced_cost, "delay_mae": balanced_delay}, "lifecycle_stage_metrics": stage}


def test_improvement_formula_and_winner_logic():
    result = calculate_improvements(metric_result(10, 20, .4, 12, 22, 11, 21, 12, 22), metric_result(8, 15, .5, 10, 18, 9, 16, 10, 17))
    assert result["cost_mae"]["absolute"] == 2
    assert result["cost_mae"]["percentage"] == 20
    assert result["risk_macro_f1"]["absolute"] == .1
    assert decide_winner(result) == "2001_2021"


def test_regression_and_mixed_winner_logic():
    regression = calculate_improvements(metric_result(8, 15, .5, 10, 18, 9, 16, 10, 17), metric_result(10, 20, .4, 12, 22, 11, 21, 12, 22))
    assert decide_winner(regression) == "2001_2019"
    mixed = calculate_improvements(metric_result(10, 20, .4, 8, 18, 9, 19, 9, 19), metric_result(8, 15, .5, 10, 20, 9, 20, 10, 20))
    assert decide_winner(mixed) == "mixed_no_clear_winner"


def test_balanced_stage_summary_is_not_row_weighted():
    stages = {
        "early": {"available": True, "cost": {"MAE": 10}, "delay": {"MAE": 20}, "risk": {"macro_f1": .5}},
        "very_late": {"available": True, "cost": {"MAE": 100}, "delay": {"MAE": 200}, "risk": {"macro_f1": .1}},
    }
    summary = _balanced_stage_summary(stages)
    assert summary["cost_mae"] == 55
    assert summary["delay_mae"] == 110


def test_common_holdout_requires_rows():
    with pytest.raises(ValueError, match="Common 2022-2025 holdout is empty"):
        make_splits(cohort().query("completion_year < 2022"))
