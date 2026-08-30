import numpy as np
import pandas as pd
from backend.app.ml.experiments.u2_isotonic_calibration_exp63 import _fit_iso

def test_isotonic_output_is_monotone_and_nonnegative_for_delay():
    n=120
    oof=pd.DataFrame({"production_prediction":np.linspace(10,100,n),"actual_delay_days":np.linspace(20,140,n)+np.sin(np.arange(n))*3,"sample_weight":np.ones(n),"lifecycle_stage":["early"]*60+["late"]*60})
    score=pd.DataFrame({"production_prediction":np.linspace(5,110,40),"lifecycle_stage":["early"]*20+["late"]*20})
    pred,details=_fit_iso(oof,score,"actual_delay_days",True)
    assert np.isfinite(pred).all() and (pred>=0).all()
    assert details["holdout_tuned"] is False

def test_global_isotonic_works_without_stage_column():
    oof=pd.DataFrame({"production_prediction":[1,2,3,4,5],"actual_cost_overrun_percentage":[0,2,2,5,7],"sample_weight":[1]*5})
    score=pd.DataFrame({"production_prediction":[1.5,3.5,4.5]})
    pred,_=_fit_iso(oof,score,"actual_cost_overrun_percentage",False)
    assert len(pred)==3 and np.isfinite(pred).all()

def test_stage_masks_are_na_safe():
    n=80
    stages=pd.Series(["early"]*30+[pd.NA]*20+["late"]*30,dtype="string")
    oof=pd.DataFrame({"production_prediction":np.linspace(1,80,n),"actual_delay_days":np.linspace(5,120,n),"sample_weight":np.ones(n),"lifecycle_stage":stages})
    score=pd.DataFrame({"production_prediction":[10,20,30,40],"lifecycle_stage":pd.Series(["early",pd.NA,"late",pd.NA],dtype="string")})
    pred,_=_fit_iso(oof,score,"actual_delay_days",True)
    assert len(pred)==4 and np.isfinite(pred).all()
