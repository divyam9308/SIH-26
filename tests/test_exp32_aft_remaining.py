import numpy as np
import pandas as pd
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.aft_remaining_exp32 import EXPERIMENT_ID,_delay_from_remaining,_remaining_frame

def test_adapter_contract(): assert get_experiment_adapter(EXPERIMENT_ID).sequence==32
def test_remaining_time_target_and_delay_reconstruction():
    frame=pd.DataFrame({"canonical_project_id":["A"],"snapshot_date":pd.to_datetime(["2020-01-01"]),"completion_date":pd.to_datetime(["2020-04-10"]),"planned_completion_date":pd.to_datetime(["2020-03-01"]),"sample_weight":[1.0]}); out=_remaining_frame(frame); assert out.exp32_remaining_days.iloc[0]==100; delay=_delay_from_remaining(out,np.array([100.0])); assert 39.0<=delay[0]<=41.0
