import numpy as np
import pandas as pd
from backend.app.ml.experiments.u6_residual_neighbors_exp67 import _neighbor_correction

def test_neighbor_model_corrects_residual_not_final_outcome():
    n=80;x=np.linspace(0,1,n)
    oof=pd.DataFrame({"production_prediction":10+10*x,"duration_ratio":1+x,"expenditure_ratio":x,"sample_weight":np.ones(n),"residual":5*np.sin(4*x)})
    score=oof.iloc[:10].drop(columns=["sample_weight","residual"]).copy()
    corr,details=_neighbor_correction(oof,score)
    assert len(corr)==10 and np.isfinite(corr).all()
    assert details["predicts_outcome_directly"] is False
    assert details["neighbors"]<=40
