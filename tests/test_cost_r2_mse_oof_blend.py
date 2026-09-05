import numpy as np
import pandas as pd

from backend.app.ml.experiments.cost_r2_mse_oof_blend import (
    ALPHA_GRID,
    select_family_and_alpha,
    weighted_metrics,
)


def _frame(actual, production):
    n = len(actual)
    return pd.DataFrame({
        "actual_cost_overrun_percentage": actual,
        "production_prediction": production,
        "sample_weight": np.ones(n),
        "oof_year": [2018] * (n // 2) + [2019] * (n - n // 2),
        "canonical_project_id": [f"p{i}" for i in range(n)],
    })


def test_weighted_metrics_r2_improves_when_squared_error_falls():
    actual = np.array([0.0, 10.0, 20.0, 100.0])
    production = np.array([0.0, 10.0, 20.0, 60.0])
    candidate = np.array([0.0, 10.0, 20.0, 80.0])
    w = np.ones(4)
    base = weighted_metrics(actual, production, w)
    better = weighted_metrics(actual, candidate, w)
    assert better["RMSE"] < base["RMSE"]
    assert better["R2"] > base["R2"]


def test_selector_minimizes_rmse_subject_to_no_mae_degradation():
    frame = _frame([0.0, 0.0, 10.0, 100.0], [0.0, 0.0, 10.0, 60.0])
    specialists = {
        "good": np.array([0.0, 0.0, 10.0, 90.0]),
        "bad": np.array([25.0, 25.0, 35.0, 130.0]),
    }
    result = select_family_and_alpha(frame, specialists)
    selected = result["selected"]
    assert selected["family"] == "good"
    assert selected["alpha"] in ALPHA_GRID
    assert selected["alpha"] > 0
    assert selected["MAE"] <= result["baseline"]["MAE"]
    assert selected["RMSE"] < result["baseline"]["RMSE"]


def test_selector_falls_back_to_zero_blend_when_specialist_hurts_mae():
    frame = _frame([0.0, 0.0, 10.0, 100.0], [0.0, 0.0, 10.0, 60.0])
    specialists = {"bad": np.array([50.0, 50.0, 60.0, 150.0])}
    result = select_family_and_alpha(frame, specialists)
    assert result["selected"]["alpha"] == 0.0
    assert result["selected"]["MAE"] == result["baseline"]["MAE"]
