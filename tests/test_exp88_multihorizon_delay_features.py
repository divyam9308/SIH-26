import pytest
from backend.app.ml.experiments.exp88_multihorizon_delay_features import EXPERIMENT_ID,EXPERIMENT_SCOPE,HORIZONS,window_contract

def test_exp88_contract():
    assert EXPERIMENT_ID=='exp_88' and EXPERIMENT_SCOPE=='delay';assert HORIZONS==(180,365,730);assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
