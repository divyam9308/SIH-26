import numpy as np
import pandas as pd

from backend.app.ml.experiments.u1_nonlinear_residual_exp62 import _fit_booster


def _frame(n=120):
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({
        "production_prediction": 100.0 + 20.0 * x,
        "cost_escalation_percentage": 5.0 + 10.0 * x,
        "schedule_slippage_days": 50.0 + 100.0 * x,
        "duration_ratio": 0.8 + 0.6 * x,
        "expenditure_ratio": 0.4 + 0.5 * x,
        "approved_cost_cr": 100.0 + 500.0 * x,
        "sample_weight": np.ones(n),
        "residual": np.sin(x * 4.0) * 8.0,
    })


def test_u1_booster_is_bounded_by_training_residuals():
    oof = _frame()
    score = _frame(30).drop(columns=["sample_weight", "residual"])
    correction, details = _fit_booster(oof, score, 62)
    assert len(correction) == len(score)
    assert np.isfinite(correction).all()
    assert np.max(np.abs(correction)) <= details["correction_cap_abs_residual_q90"] + 1e-9
    assert "production_prediction" in details["features"]


def test_u1_booster_ignores_unavailable_optional_columns():
    oof = _frame().drop(columns=["approved_cost_cr"])
    score = _frame(10).drop(columns=["approved_cost_cr", "sample_weight", "residual"])
    correction, details = _fit_booster(oof, score, 63)
    assert len(correction) == 10
    assert "approved_cost_cr" not in details["features"]
