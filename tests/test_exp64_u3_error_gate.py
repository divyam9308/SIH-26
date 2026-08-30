import numpy as np
import pandas as pd
from backend.app.ml.experiments.u3_error_risk_gate_exp64 import _fit_gate

def test_error_gate_returns_bounded_sparse_correction():
    n=160;x=np.linspace(0,1,n)
    oof=pd.DataFrame({"production_prediction":100+20*x,"cost_escalation_percentage":10*x,"schedule_slippage_days":200*x,"duration_ratio":1+x,"expenditure_ratio":x,"sample_weight":np.ones(n),"residual":np.where(x>.75,25*np.sin(x*10),2*np.sin(x*10))})
    score=oof.iloc[:40].drop(columns=["sample_weight","residual"]).copy()
    corr,details=_fit_gate(oof,score,64)
    assert len(corr)==40 and np.isfinite(corr).all()
    assert details["gate_threshold"]==0.5 and details["holdout_tuned"] is False
    assert np.max(np.abs(corr))<=details["correction_cap_q90"]+1e-9
