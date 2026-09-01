import pandas as pd
import pytest
from backend.app.ml.experiments.exp118_fiscal_schedule_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,_engineer,window_contract

def test_exp118_fiscal_calendar_contract():
    assert EXPERIMENT_ID=='exp_118' and EXPERIMENT_SCOPE=='delay'
    f=pd.DataFrame({'canonical_project_id':['A','A'],'snapshot_date':['2020-04-01','2021-02-01'],'expenditure_ratio':[.2,.5],'schedule_slippage_days':[0,10],'progress_deviation':[0,-5]})
    x=_engineer(f)
    assert list(x['exp118_fiscal_month'])==[1.0,11.0]
    assert list(x['exp118_fiscal_q4'])==[0.0,1.0]
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
