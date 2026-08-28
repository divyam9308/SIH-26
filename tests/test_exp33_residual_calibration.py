import numpy as np
import pandas as pd
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.residual_calibration_exp33 import EXPERIMENT_ID,_corrections,_weighted_median

def test_adapter_contract():assert get_experiment_adapter(EXPERIMENT_ID).sequence==33
def test_weighted_median_and_calibration_fallbacks():
    assert _weighted_median([0,10,20],[1,10,1])==10.0
    cal={"edges":[-np.inf,5.0,np.inf],"global_median":1.0,"bin_medians":{0:2.0,1:3.0},"stage_bin_medians":{("early",0):4.0}}; frame=pd.DataFrame({"lifecycle_stage":["early","late"]}); corr=_corrections(frame,np.array([2.0,8.0]),cal); assert corr.tolist()==[4.0,3.0]
