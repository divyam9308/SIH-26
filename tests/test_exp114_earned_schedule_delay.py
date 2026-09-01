import pandas as pd
import pytest
from backend.app.ml.experiments.exp114_earned_schedule_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,_engineer,window_contract

def test_exp114_contract_and_prefix_formula():
    assert EXPERIMENT_ID=='exp_114' and EXPERIMENT_SCOPE=='delay'
    f=pd.DataFrame({'physical_progress':[50.0],'elapsed_duration_days':[100.0],'planned_duration_days':[200.0],'snapshot_date':['2020-01-01'],'planned_completion_date':['2020-04-10'],'schedule_slippage_days':[0.0]})
    x=_engineer(f)
    assert x['exp114_implied_total_duration'].iloc[0]==200.0
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
