import pytest
from backend.app.ml.experiments.exp87_tail_specialist_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,GATE,SCALE,window_contract

def test_exp87_contract():
    assert EXPERIMENT_ID=='exp_87' and EXPERIMENT_SCOPE=='delay';assert GATE>=.7 and 0<SCALE<=.5;assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
