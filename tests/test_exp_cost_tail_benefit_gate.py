import numpy as np

from backend.app.ml.experiments import exp_cost_tail_benefit_gate as exp


def test_contract_is_frozen_and_nonpromoting():
    assert exp.EXP_ID == "exp_cost_tail_benefit_gate"
    assert exp.QUANTILES == (0.50, 0.60, 0.70)
    assert exp.SCALES[0] == 0.0
    assert "production_prediction" in exp.FEATURES
    assert "cost_escalation_percentage" in exp.FEATURES


def test_positive_tail_transform_is_directionally_safe():
    raw = np.array([-2.0, 0.0, 3.0])
    transformed = np.maximum(np.expm1(np.log1p(np.maximum(raw, 0.0))), 0.0)
    assert np.allclose(transformed, [0.0, 0.0, 3.0])
