import pandas as pd
from backend.app.ml.experiments.u10_robust_trends_exp71 import TREND_FEATURES,enrich_robust_trends

def _base():
    return pd.DataFrame({"canonical_project_id":["p"]*4,"snapshot_date":pd.to_datetime(["2020-01-01","2020-04-01","2020-07-01","2020-10-01"]),"cost_escalation_percentage":[0.,3.,7.,9.],"expenditure_ratio":[.1,.2,.4,.55]})

def test_future_row_does_not_change_earlier_robust_trends():
    base=_base();a=enrich_robust_trends(base)[TREND_FEATURES].reset_index(drop=True)
    future=pd.concat([base,pd.DataFrame({"canonical_project_id":["p"],"snapshot_date":pd.to_datetime(["2021-01-01"]),"cost_escalation_percentage":[1000.],"expenditure_ratio":[10.]})],ignore_index=True)
    b=enrich_robust_trends(future).iloc[:4][TREND_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)

def test_multi_resolution_features_are_created():
    out=enrich_robust_trends(_base())
    assert set(TREND_FEATURES).issubset(out.columns)
    assert out["exp71_cost_escalation_percentage_slope_12m"].notna().any()
