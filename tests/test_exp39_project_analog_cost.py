import numpy as np
from backend.app.ml.experiments import adapter_exp39
from backend.app.ml.experiments.project_analog_cost_exp39 import _weighted_median


def test_exp39_adapter_contract():
    assert adapter_exp39.EXPERIMENT_ID == "exp_39"
    assert adapter_exp39.EXPERIMENT_SEQUENCE == 39
    assert adapter_exp39.EXPERIMENT_SCOPE == "cost"


def test_weighted_median_prefers_majority_weight():
    values = np.array([1.0, 10.0, 100.0])
    weights = np.array([1.0, 8.0, 1.0])
    assert _weighted_median(values, weights) == 10.0
