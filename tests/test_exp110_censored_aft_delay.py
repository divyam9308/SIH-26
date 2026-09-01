import pytest
from backend.app.ml.experiments.exp110_censored_aft_delay import CENSOR_DAYS,EXPERIMENT_ID,EXPERIMENT_SCOPE,window_contract

def test_exp110_contract():
    assert EXPERIMENT_ID=='exp_110' and EXPERIMENT_SCOPE=='delay'
    assert CENSOR_DAYS==365.0
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
