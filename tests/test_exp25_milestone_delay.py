from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.milestone_delay_exp25 import (
    add_milestone_features,
    decision,
    enrich_with_monthly_milestones,
)


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


def test_supervised_snapshot_uses_full_monthly_history() -> None:
    supervised = pd.DataFrame({
        "canonical_project_id": ["P1"],
        "snapshot_date": ["2020-04-01"],
    })
    monthly = pd.DataFrame({
        "canonical_project_id": ["P1", "P1", "P1", "P1"],
        "snapshot_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "milestone_status": ["2/10", "2/10", "3/10", "4/10"],
    })
    out = enrich_with_monthly_milestones(supervised, history=monthly).iloc[0]
    assert out.exp25_milestone_delta == 1
    assert out.exp25_milestone_velocity > 0


def test_future_row_cannot_change_earlier_features() -> None:
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


def test_delay_only_promotion_rule() -> None:
    assert decision(0.01) == "PROMOTION CANDIDATE"
    assert decision(0.0) == "REGRESSION / DO NOT PROMOTE"
    assert decision(-0.1) == "REGRESSION / DO NOT PROMOTE"
