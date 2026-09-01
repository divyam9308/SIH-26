import pytest
from backend.app.ml.experiments.exp115_agency_backlog_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,PORTFOLIO_FEATURES,window_contract

def test_exp115_contract():
    assert EXPERIMENT_ID=='exp_115' and EXPERIMENT_SCOPE=='delay'
    assert 'exp115_backlog_throughput_pressure' in PORTFOLIO_FEATURES
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
