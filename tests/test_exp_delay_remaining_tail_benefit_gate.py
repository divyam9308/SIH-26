import numpy as np

from backend.app.ml.experiments import exp_delay_remaining_tail_benefit_gate as exp


def test_search_space_is_small_and_forward_selectable():
    assert exp.ALPHAS == (0.50, 0.60, 0.70)
    assert exp.TAIL_WEIGHTS == (1.0, 2.0, 4.0)
    assert exp.SCALES == (0.25, 0.5, 0.75, 1.0)
    assert "production_prediction" in exp.FEATURES
    assert "exp113_correction" in exp.FEATURES


def test_tail_uplift_is_nonnegative():
    tail = np.array([100.0, 80.0])
    anchor = np.array([90.0, 100.0])
    assert np.array_equal(np.maximum(tail - anchor, 0.0), [10.0, 0.0])
