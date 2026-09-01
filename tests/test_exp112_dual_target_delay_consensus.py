import pytest
from backend.app.ml.experiments.exp112_dual_target_delay_consensus import EXPERIMENT_ID,EXPERIMENT_SCOPE,GRID,window_contract

def test_exp112_contract():
    assert EXPERIMENT_ID=='exp_112' and EXPERIMENT_SCOPE=='delay'
    assert 0.0 in GRID and 1.0 in GRID
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
