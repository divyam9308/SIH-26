from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.milestone_trajectory_exp22 import (
    _decision,
    add_milestone_features,
    enrich_with_monthly_milestones,
)


def test_milestone_parser_and_past_only_velocity() -> None:
    frame = pd.DataFrame({
        "canonical_project_id": ["P1", "P1", "P1"],
        "snapshot_date": ["2020-01-01", "2020-04-01", "2020-07-01"],
        "milestone_status": ["2/10", "4/10", "4/10"],
    })
    out = add_milestone_features(frame)
    assert out.loc[0, "exp22_milestone_ratio"] == 0.2
    assert out.loc[1, "exp22_milestone_delta"] == 2
    assert out.loc[1, "exp22_milestone_velocity"] > 0
    assert out.loc[2, "exp22_milestone_stagnant"] == 1
    assert out.loc[2, "exp22_months_since_milestone_change"] > 0


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
    assert out.exp22_milestone_delta == 1
    assert out.exp22_milestone_velocity > 0


def test_promotion_verdict_requires_both_targets_nonworse() -> None:
    assert _decision(0.2, 0.1) == "PROMOTION CANDIDATE"
    assert _decision(-0.1, 1.0) == "REGRESSION / DO NOT PROMOTE"
