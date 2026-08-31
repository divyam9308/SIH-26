import pytest
from backend.app.ml.experiments.exp81_delay_informed_cost import EXPERIMENT_ID,EXPERIMENT_SCOPE,window_contract

def test_exp81_contract():
    assert EXPERIMENT_ID=='exp_81' and EXPERIMENT_SCOPE=='cost';assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
