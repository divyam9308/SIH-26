import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp141_stage_conditional_cost import (
    STAGE_FEATURES,
    STAGES,
    _fit_stage_conditional_residual_layer,
    _normalize_stage,
    _stage_metrics,
)


def test_normalize_stage_maps_properly():
    s = pd.Series(["Early", "MID", "LATE", None, "VERY_LATE"])
    norm = _normalize_stage(s)
    assert norm.tolist() == ["early", "mid", "late", "mid", "very_late"]


def test_stage_metrics_computation():
    frame = pd.DataFrame(
        {
            "lifecycle_stage": ["early", "early", "mid", "mid", "late", "late", "very_late", "very_late"],
            "actual_cost_overrun_percentage": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
            "sample_weight": [1.0] * 8,
            "canonical_project_id": [f"proj_{i}" for i in range(8)],
        }
    )
    pred = np.array([9.0, 14.0, 21.0, 24.0, 29.0, 36.0, 38.0, 48.0])
    metrics = _stage_metrics(frame, pred)

    for stage in STAGES:
        assert stage in metrics
        assert metrics[stage]["available"] is True
        assert "MAE" in metrics[stage]
        assert "R2" in metrics[stage]


def test_fit_stage_conditional_residual_layer_mock():
    rng = np.random.RandomState(141)
    n = 160
    stages_cycle = ["early", "mid", "late", "very_late"] * (n // 4)
    oof = pd.DataFrame(
        {
            "oof_year": [2018] * 40 + [2019] * 40 + [2020] * 40 + [2021] * 40,
            "lifecycle_stage": stages_cycle,
            "production_prediction": rng.uniform(10, 50, n),
            "actual_cost_overrun_percentage": rng.uniform(5, 60, n),
            "sample_weight": np.ones(n),
            "approved_cost_cr": rng.uniform(100, 1000, n),
            "cost_escalation_percentage": rng.uniform(0, 30, n),
            "expenditure_ratio": rng.uniform(0.1, 1.1, n),
            "duration_ratio": rng.uniform(0.8, 1.8, n),
            "progress_deviation": rng.uniform(-15, 15, n),
            "cost_growth_velocity_6m": rng.uniform(-0.1, 0.2, n),
            "schedule_slippage_days": rng.uniform(0, 500, n),
            "physical_progress": rng.uniform(10, 100, n),
        }
    )
    score = oof.iloc[:20].copy()
    correction, details = _fit_stage_conditional_residual_layer(oof, score, seed=141, min_stage_rows=5)

    assert len(correction) == len(score)
    assert "stage_scales" in details
    assert set(details["stage_scales"].keys()) == set(STAGES)
