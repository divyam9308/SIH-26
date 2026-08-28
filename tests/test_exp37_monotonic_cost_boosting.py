from backend.app.ml.experiments import adapter_exp37
from backend.app.ml.experiments.monotonic_cost_boosting_exp37 import _constraints


def test_exp37_adapter_contract():
    assert adapter_exp37.EXPERIMENT_ID == "exp_37"
    assert adapter_exp37.EXPERIMENT_SEQUENCE == 37
    assert adapter_exp37.EXPERIMENT_SCOPE == "cost"


def test_monotonic_constraint_vector_is_explicit():
    cols = ["cost_escalation_percentage", "sector_Rail", "duration_ratio"]
    assert _constraints(cols) == (1, 0, 1)
