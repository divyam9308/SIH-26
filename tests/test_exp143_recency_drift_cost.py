import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp143_recency_drift_cost import (
    EXP143_FEATURES,
    calculate_recency_weights,
)


def test_calculate_recency_weights_monotonicity():
    years = pd.Series([2005, 2010, 2015, 2020, 2022])
    base_weights = np.ones(5)
    weights = calculate_recency_weights(years, base_weights, decay_lambda=1.2)

    assert len(weights) == 5
    assert np.all(weights > 0.0)
    for i in range(len(weights) - 1):
        assert weights[i] <= weights[i + 1]


def test_recency_weights_normalized_mean():
    years = pd.Series(np.random.randint(2005, 2023, 100))
    base_weights = np.random.uniform(0.5, 2.0, 100)
    weights = calculate_recency_weights(years, base_weights, decay_lambda=1.5)

    assert np.isclose(np.mean(weights), 1.0, atol=1e-5)
