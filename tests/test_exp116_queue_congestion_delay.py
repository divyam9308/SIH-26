import pytest
from backend.app.ml.experiments.exp116_queue_congestion_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,QUEUE_FEATURES,window_contract

def test_exp116_contract():
    assert EXPERIMENT_ID=='exp_116' and EXPERIMENT_SCOPE=='delay'
    assert 'exp116_agency_queue_pressure' in QUEUE_FEATURES
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
