import pandas as pd
import pytest
from backend.app.ml.experiments.exp117_monsoon_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,_engineer,window_contract

def test_exp117_calendar_proxy_contract():
    assert EXPERIMENT_ID=='exp_117' and EXPERIMENT_SCOPE=='delay'
    f=pd.DataFrame({'snapshot_date':['2020-07-01','2020-12-01'],'sector':['Roads','Roads'],'schedule_slippage_days':[10,10],'progress_deviation':[-5,-5]})
    x=_engineer(f)
    assert list(x['exp117_monsoon'])==[1.0,0.0]
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
