import pandas as pd,pytest
from backend.app.ml.experiments.exp85_revision_shock_delay import EXPERIMENT_ID,EXPERIMENT_SCOPE,engineer,window_contract

def test_exp85_contract():
    assert EXPERIMENT_ID=='exp_85' and EXPERIMENT_SCOPE=='delay';assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)

def test_schedule_shocks_prefix_only():
    a=pd.DataFrame({'canonical_project_id':['p','p'],'snapshot_date':['2020-01-01','2020-02-01'],'revised_completion_date':['2021-01-01','2021-03-01']});x=engineer(a);b=pd.concat([a,pd.DataFrame({'canonical_project_id':['p'],'snapshot_date':['2020-03-01'],'revised_completion_date':['2022-01-01']})],ignore_index=True);y=engineer(b);cols=['exp85_schedule_revision_days','exp85_schedule_revision_count','exp85_max_extension_days'];pd.testing.assert_frame_equal(x[cols].reset_index(drop=True),y.iloc[:2][cols].reset_index(drop=True))
