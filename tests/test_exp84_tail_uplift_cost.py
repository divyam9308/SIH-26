import pytest
from backend.app.ml.experiments.exp84_tail_uplift_cost import EXPERIMENT_ID,EXPERIMENT_SCOPE,GATE,SCALE,window_contract

def test_exp84_contract():
    assert EXPERIMENT_ID=='exp_84' and EXPERIMENT_SCOPE=='cost';assert GATE>=.8 and 0<SCALE<=.25;assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
