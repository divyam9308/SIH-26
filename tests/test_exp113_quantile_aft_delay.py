import pytest
from backend.app.ml.experiments.exp113_quantile_aft_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,QUANTILES,window_contract

def test_exp113_contract():
    assert EXPERIMENT_ID=='exp_113' and EXPERIMENT_SCOPE=='delay'
    assert QUANTILES==(.25,.5,.75)
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
