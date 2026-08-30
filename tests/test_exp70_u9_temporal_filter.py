import numpy as np
import pandas as pd
from backend.app.ml.experiments.u9_temporal_consistency_exp70 import causal_completion_filter

def test_future_prediction_cannot_change_earlier_filtered_values():
    base=pd.DataFrame({"canonical_project_id":["p"]*3,"snapshot_date":pd.to_datetime(["2020-01-01","2020-02-01","2020-03-01"]),"planned_completion_date":pd.to_datetime(["2021-01-01"]*3)})
    a=causal_completion_filter(base,np.array([100.,200.,150.]),.5)
    future=pd.concat([base,pd.DataFrame({"canonical_project_id":["p"],"snapshot_date":pd.to_datetime(["2020-04-01"]),"planned_completion_date":pd.to_datetime(["2021-01-01"])})],ignore_index=True)
    b=causal_completion_filter(future,np.array([100.,200.,150.,5000.]),.5)
    np.testing.assert_allclose(a,b[:3])

def test_alpha_one_is_identity_when_planned_date_is_fixed():
    frame=pd.DataFrame({"canonical_project_id":["p","p"],"snapshot_date":pd.to_datetime(["2020-01-01","2020-02-01"]),"planned_completion_date":pd.to_datetime(["2021-01-01","2021-01-01"])})
    raw=np.array([120.,180.])
    np.testing.assert_allclose(causal_completion_filter(frame,raw,1.0),raw)
