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

def test_neighbor_model_imputes_training_only_missing_values():
    n=50;x=np.linspace(0,1,n)
    oof=pd.DataFrame({"production_prediction":10+x,"duration_ratio":[np.nan]*n,"expenditure_ratio":x,"sample_weight":np.ones(n),"residual":np.cos(x)})
    score=pd.DataFrame({"production_prediction":[10.1,np.nan,10.8],"duration_ratio":[np.nan,np.nan,np.nan],"expenditure_ratio":[.1,np.inf,.8]})
    corr,details=_neighbor_correction(oof,score)
    assert np.isfinite(corr).all()
    assert details["training_medians"]["duration_ratio"]==0.0
    assert details["training_medians"]["production_prediction"]>0
