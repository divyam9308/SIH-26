from backend.app.ml.experiments.combined_exp51_exp58_exp61 import (
    CHANGED_DIMENSION,
    EXPERIMENT_ID,
    EXPERIMENT_SCOPE,
)


def test_exp61_is_isolated_target_composition():
    assert EXPERIMENT_ID == "exp_61"
    assert EXPERIMENT_SCOPE == "cost+delay"
    assert CHANGED_DIMENSION == "exp51_cost_plus_exp58_delay"
