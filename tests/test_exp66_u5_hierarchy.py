import numpy as np
import pandas as pd
from backend.app.ml.experiments.u5_hierarchical_residual_exp66 import _fit_hierarchy

def test_hierarchical_residual_correction_is_training_only_and_finite():
    oof=pd.DataFrame({"residual":[1.,2.,10.,12.],"sample_weight":[1.,1.,1.,1.],"lifecycle_stage":["early","early","late","late"],"project_size_category":["S","S","L","L"],"_norm_sector":["road","road","rail","rail"],"_norm_implementing_agency":["a","a","b","b"]})
    score=oof.drop(columns=["residual","sample_weight"]).copy()
    corr,details=_fit_hierarchy(oof,score)
    assert len(corr)==4 and np.isfinite(corr).all()
    assert details["holdout_tuned"] is False
    assert np.max(np.abs(corr))<=details["correction_cap_q90"]+1e-9
