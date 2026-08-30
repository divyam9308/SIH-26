import numpy as np
import pandas as pd
from backend.app.ml.experiments.u11_u1_u3_delay_blend_exp72 import _weighted_mae,select_u1_weight

def test_weighted_mae_prefers_better_prediction():
    a=np.array([0.,10.]);w=np.array([1.,1.])
    assert _weighted_mae(a,np.array([0.,9.]),w) < _weighted_mae(a,np.array([4.,4.]),w)

def test_selection_fallback_is_training_only_and_bounded():
    rows=[]
    for year in (2017,2018):
        for i in range(20):
            rows.append({'oof_year':year,'production_prediction':100.+i,'actual_delay_days':105.+i,'sample_weight':1.,'residual':5.,'canonical_project_id':f'{year}-{i}','cost_escalation_percentage':0.,'schedule_slippage_days':0.,'duration_ratio':1.,'expenditure_ratio':.5,'progress_deviation':0.,'approved_cost_cr':100.})
    w,details=select_u1_weight(pd.DataFrame(rows))
    assert 0.0 <= w <= 1.0
    assert details['selection_years']==[]
    assert details['fallback_weight']==0.5
