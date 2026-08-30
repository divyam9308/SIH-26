import numpy as np
import pandas as pd
from backend.app.ml.experiments.u12_u1_u10_cost_blend_exp73 import _median_pairwise_slope,_weighted_mae,select_weights

def test_median_pairwise_slope_is_finite_for_linear_prefix():
    assert np.isclose(_median_pairwise_slope([0.,30.,60.],[0.,10.,20.]),10.145833333333334,rtol=1e-6)

def test_weighted_mae_prefers_closer_prediction():
    a=np.array([0.,10.]);w=np.array([1.,1.])
    assert _weighted_mae(a,np.array([0.,9.]),w) < _weighted_mae(a,np.array([5.,5.]),w)

def test_weight_selection_fallback_respects_production_floor():
    rows=[]
    for year in (2017,2018):
        for i in range(20):
            rows.append({'oof_year':year,'production_prediction':10.+i,'actual_cost_overrun_percentage':12.+i,'sample_weight':1.,'residual':2.,'canonical_project_id':f'{year}-{i}','snapshot_date':pd.Timestamp(f'{year}-01-01'),'cost_escalation_percentage':0.,'schedule_slippage_days':0.,'duration_ratio':1.,'expenditure_ratio':.5,'progress_deviation':0.,'approved_cost_cr':100.})
    weights,details=select_weights(pd.DataFrame(rows),pd.DataFrame(columns=['canonical_project_id','snapshot_date','oof_year','u10_prediction']))
    assert weights==(0.5,0.25,0.25)
    assert details['fallback_weights']['production']>=0.5
