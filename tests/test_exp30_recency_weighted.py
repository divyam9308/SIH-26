import numpy as np
import pandas as pd
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.recency_weighted_exp30 import EXPERIMENT_ID,_apply_recency_weights

def test_adapter_contract(): assert get_experiment_adapter(EXPERIMENT_ID).sequence==30
def test_recency_weights_favor_newer_projects_without_changing_baseline():
    frame=pd.DataFrame({"completion_year":[2010,2015,2020],"sample_weight":[1.0,1.0,1.0]})
    baseline=_apply_recency_weights(frame,2020,None); assert np.allclose(baseline.sample_weight,frame.sample_weight)
    weighted=_apply_recency_weights(frame,2020,4.0); assert weighted.sample_weight.iloc[2]>weighted.sample_weight.iloc[1]>weighted.sample_weight.iloc[0]; assert np.isclose(weighted.sample_weight.sum(),3.0)
