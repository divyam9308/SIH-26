import numpy as np
import pandas as pd

from backend.app.ml.experiments import exp_cost_dynamic_exp105_scale as exp


def test_multiplier_grid_and_labels():
    frame = pd.DataFrame({
        "actual_cost_overrun_percentage": [10.0, 20.0],
        "production_base": [0.0, 0.0],
        "exp105_correction": [10.0, 10.0],
    })
    labels = exp._labels(frame)
    assert exp.MULTIPLIERS.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert np.array_equal(labels, [2, 4])


def test_router_uses_existing_correction_only():
    assert "production_base" in exp.FEATURES
    assert "exp105_correction" in exp.FEATURES
