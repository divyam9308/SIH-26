import numpy as np
import pandas as pd

from backend.app.ml.experiments.hybrid_cost_regime import (
    REGIMES,
    confidence_alpha,
    cost_regime,
    hard_route,
    soft_route,
)
from backend.app.ml.real_time_windows import FEATURES, TARGET_COLUMNS


def test_regime_boundaries():
    assert cost_regime(-10) == "COST_SAVING"
    assert cost_regime(0) == "COST_SAVING"
    assert cost_regime(10) == "LOW"
    assert cost_regime(20) == "LOW"
    assert cost_regime(50) == "MEDIUM"
    assert cost_regime(100) == "MEDIUM"
    assert cost_regime(150) == "HIGH"
    assert cost_regime(200) == "HIGH"
    assert cost_regime(300) == "EXTREME"


def test_exact_production_feature_contract_is_used():
    assert FEATURES == [
        "approved_cost_cr",
        "sector_average_delay",
        "sector_average_cost_overrun",
        "sector",
        "project_size_category",
    ]


def test_no_target_or_actual_regime_is_a_feature():
    assert "actual_cost_overrun_percentage" not in FEATURES
    assert "cost_regime" not in FEATURES
    assert not set(FEATURES).intersection(TARGET_COLUMNS)


def test_probabilities_and_soft_weights_are_well_formed():
    probabilities = np.array([[0.05, 0.10, 0.55, 0.25, 0.05]])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    experts = np.array([[-15.0, 12.0, 58.0, 145.0, 320.0]])
    global_predictions = np.array([40.0])
    result = soft_route(probabilities, experts, global_predictions)
    assert result.shape == (1,)
    assert np.isfinite(result[0])


def test_routing_never_accepts_actual_outcome_or_actual_regime():
    probabilities = np.array([[0.05, 0.10, 0.55, 0.25, 0.05]])
    experts = np.array([[-15.0, 12.0, 58.0, 145.0, 320.0]])
    global_predictions = np.array([40.0])
    hard = hard_route(probabilities, experts, global_predictions)
    soft = soft_route(probabilities, experts, global_predictions)
    assert hard.shape == soft.shape == (1,)


def test_low_classifier_confidence_falls_back_to_global_prediction():
    probabilities = np.full((1, len(REGIMES)), 1.0 / len(REGIMES))
    experts = np.array([[-20.0, 10.0, 50.0, 150.0, 300.0]])
    global_predictions = np.array([27.5])
    assert np.allclose(confidence_alpha(probabilities), 0.0)
    assert np.allclose(hard_route(probabilities, experts, global_predictions), global_predictions)
    assert np.allclose(soft_route(probabilities, experts, global_predictions), global_predictions)


def test_high_classifier_confidence_uses_expert_signal():
    probabilities = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    experts = np.array([[-20.0, 10.0, 60.0, 150.0, 300.0]])
    global_predictions = np.array([27.5])
    assert np.allclose(confidence_alpha(probabilities), 1.0)
    assert np.allclose(hard_route(probabilities, experts, global_predictions), [60.0])
    assert np.allclose(soft_route(probabilities, experts, global_predictions), [60.0])
