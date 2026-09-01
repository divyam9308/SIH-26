import pandas as pd
import pytest
from backend.app.ml.experiments.exp119_state_space_slippage_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,_engineer,window_contract

def test_exp119_state_filter_contract():
    assert EXPERIMENT_ID=='exp_119' and EXPERIMENT_SCOPE=='delay'
    f=pd.DataFrame({'canonical_project_id':['A','A'],'snapshot_date':['2020-01-01','2020-02-01'],'schedule_slippage_days':[10.0,20.0]})
    x=_engineer(f)
    assert x['exp119_slip_level'].notna().all()
    assert x['exp119_slip_projection_3'].iloc[1] >= 0
    assert window_contract(2019)[:2]==(2020,2025)
    assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)
