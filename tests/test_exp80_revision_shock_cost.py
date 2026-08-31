import pandas as pd,pytest
from backend.app.ml.experiments.exp80_revision_shock_cost import EXPERIMENT_ID,EXPERIMENT_SCOPE,engineer,window_contract

def test_contract_only_standard_windows():
    assert EXPERIMENT_ID=='exp_80' and EXPERIMENT_SCOPE=='cost';assert window_contract(2019)[:2]==(2020,2025);assert window_contract(2021)[:2]==(2022,2025)
    with pytest.raises(ValueError): window_contract(2022)

def test_revision_features_are_prefix_only():
    base=pd.DataFrame({'canonical_project_id':['p','p'],'snapshot_date':['2020-01-01','2020-02-01'],'revised_cost_cr':[100.,110.],'approved_cost_cr':[100.,100.]})
    a=engineer(base);b=engineer(pd.concat([base,pd.DataFrame({'canonical_project_id':['p'],'snapshot_date':['2020-03-01'],'revised_cost_cr':[150.],'approved_cost_cr':[100.]})],ignore_index=True))
    cols=['exp80_revision_shock_pct','exp80_revision_count','exp80_max_abs_shock_pct','exp80_days_since_revision'];pd.testing.assert_frame_equal(a[cols].reset_index(drop=True),b.iloc[:2][cols].reset_index(drop=True))
