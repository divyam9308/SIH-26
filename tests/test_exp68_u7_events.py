import pandas as pd
from backend.app.ml.experiments.u7_event_trajectory_exp68 import EVENT_FEATURES,enrich_events

def _base():
    return pd.DataFrame({"canonical_project_id":["p"]*3,"snapshot_date":pd.to_datetime(["2020-01-01","2020-02-01","2020-03-01"]),"cost_escalation_percentage":[0.,8.,9.],"schedule_slippage_days":[0.,40.,45.]})

def test_future_row_cannot_change_earlier_event_features():
    base=_base();a=enrich_events(base)[EVENT_FEATURES].reset_index(drop=True)
    future=pd.concat([base,pd.DataFrame({"canonical_project_id":["p"],"snapshot_date":pd.to_datetime(["2020-04-01"]),"cost_escalation_percentage":[1000.],"schedule_slippage_days":[2000.]})],ignore_index=True)
    b=enrich_events(future).iloc[:3][EVENT_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)

def test_event_thresholds_count_material_changes():
    out=enrich_events(_base())
    assert out["exp68_cost_shock_count"].iloc[-1]==1
    assert out["exp68_schedule_shock_count"].iloc[-1]==1
