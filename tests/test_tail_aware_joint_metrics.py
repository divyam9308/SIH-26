import numpy as np
import pandas as pd

from backend.app.ml.experiments.tail_aware_joint_metrics import (
    COST_FEATURES,
    apply_tail_aware_layer,
    fit_tail_aware_layer,
    metrics,
)


def _frame():
    rows = []
    for year in (2017, 2018, 2019, 2020, 2021):
        for i in range(50):
            anchor = float(i)
            actual = anchor + (10.0 if i >= 45 else 2.0)
            rows.append({
                "oof_year": year,
                "production_prediction": anchor,
                "actual_cost_overrun_percentage": actual,
                "sample_weight": 1.0,
                "cost_escalation_percentage": float(i),
                "expenditure_ratio": i / 50.0,
                "progress_deviation": -float(i) / 10.0,
                "schedule_slippage_days": float(i * 2),
                "duration_ratio": 1.0 + i / 100.0,
                "physical_progress": float(i),
                "approved_cost_cr": 100.0 + i,
                "planned_duration_days": 1000.0,
                "elapsed_duration_days": 500.0 + i,
            })
    return pd.DataFrame(rows)


def test_tail_aware_layer_uses_training_quantiles_and_forward_years():
    frame = _frame()
    layer = fit_tail_aware_layer(
        frame,
        actual_col="actual_cost_overrun_percentage",
        features=COST_FEATURES,
        target="cost",
        seed=19501,
        nonnegative=False,
    )
    assert layer.p95 > layer.p90
    assert layer.scale in {0.0, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0}
    pred = apply_tail_aware_layer(layer, frame, frame["production_prediction"].to_numpy(float), nonnegative=False)
    assert len(pred) == len(frame)


def test_metrics_reports_mae_rmse_and_r2():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([0.0, 9.0, 18.0])
    m = metrics(y, p, np.ones(3))
    assert set(m) == {"MAE", "RMSE", "R2"}
    assert m["MAE"] > 0
    assert m["RMSE"] >= m["MAE"]
    assert m["R2"] < 1.0
