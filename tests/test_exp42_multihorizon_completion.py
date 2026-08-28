import numpy as np
from backend.app.ml.experiments import adapter_exp42
from backend.app.ml.experiments.multihorizon_completion_exp42 import monotonicize


def test_exp42_adapter_contract():
    assert adapter_exp42.EXPERIMENT_ID == "exp_42"
    assert adapter_exp42.EXPERIMENT_SEQUENCE == 42
    assert adapter_exp42.EXPERIMENT_SCOPE == "delay"


def test_probabilities_are_monotonic_by_horizon():
    p = np.array([[0.2, 0.6, 0.5, 0.9]])
    out = monotonicize(p)
    assert out.tolist() == [[0.2, 0.6, 0.6, 0.9]]
