import pytest
from backend.app.ml.experiments.exp111_normalized_remaining_aft_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,window_contract

def test_exp111_contract():
    assert EXPERIMENT_ID=='exp_111' and EXPERIMENT_SCOPE=='delay'
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
