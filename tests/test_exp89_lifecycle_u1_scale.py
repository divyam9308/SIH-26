import pytest
from backend.app.ml.experiments.exp89_lifecycle_u1_scale import EXPERIMENT_ID,EXPERIMENT_SCOPE,GRID,window_contract

def test_exp89_contract():
    assert EXPERIMENT_ID=='exp_89' and EXPERIMENT_SCOPE=='delay';assert min(GRID)==0 and max(GRID)==1;assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
