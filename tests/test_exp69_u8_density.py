import numpy as np
import pandas as pd
from backend.app.ml.experiments.u8_density_ratio_exp69 import density_project_weights

def test_density_weights_are_training_only_and_clipped():
    n=40
    frame=pd.DataFrame({"canonical_project_id":[f"p{i}" for i in range(n)],"snapshot_date":pd.to_datetime(["2020-01-01"]*n),"completion_year":[2005+i%17 for i in range(n)],"approved_cost_cr":np.linspace(10,1000,n),"duration_ratio":np.linspace(.5,2,n),"cost_escalation_percentage":np.linspace(-5,50,n),"schedule_slippage_days":np.linspace(0,500,n),"expenditure_ratio":np.linspace(.1,1.2,n),"progress_deviation":np.linspace(-30,20,n)})
    mapping,details=density_project_weights(frame,2021)
    vals=np.array(list(mapping.values()))
    assert len(vals)==n and np.isfinite(vals).all()
    assert vals.min()>=.25-1e-9 and vals.max()<=4+1e-9
    assert details["future_holdout_used"] is False
