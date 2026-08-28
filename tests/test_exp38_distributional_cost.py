from backend.app.ml.experiments import adapter_exp38


def test_exp38_adapter_contract():
    assert adapter_exp38.EXPERIMENT_ID == "exp_38"
    assert adapter_exp38.EXPERIMENT_SEQUENCE == 38
    assert adapter_exp38.EXPERIMENT_SCOPE == "cost"
    assert callable(adapter_exp38.fit_against_production)
