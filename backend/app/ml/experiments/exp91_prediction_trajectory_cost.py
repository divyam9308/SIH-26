"""Exp91: production Cost prediction trajectory meta-features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_91';EXPERIMENT_NAME='Production-prediction trajectory Cost meta model';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=91

def _trajectory(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x=x.sort_values(['canonical_project_id','snapshot_date']).copy();g=x.groupby('canonical_project_id',sort=False)['production_prediction'];x['exp91_pred_change_1']=g.diff();x['exp91_pred_change_3']=x['production_prediction']-g.shift(3);x['exp91_pred_change_6']=x['production_prediction']-g.shift(6);x['exp91_pred_vol_3']=g.transform(lambda s:s.rolling(3,min_periods=2).std());x['exp91_pred_vol_6']=g.transform(lambda s:s.rolling(6,min_periods=2).std());x['exp91_pred_accel']=x['exp91_pred_change_1']-x.groupby('canonical_project_id',sort=False)['exp91_pred_change_1'].shift(1);x['exp91_pred_max_jump_6']=x.groupby('canonical_project_id',sort=False)['exp91_pred_change_1'].transform(lambda s:s.abs().rolling(6,min_periods=1).max());return x.sort_index()
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=_trajectory(current_cost_oof(ctx['train'],ctx['cost_model']));score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];score=_trajectory(score);features=['production_prediction','exp91_pred_change_1','exp91_pred_change_3','exp91_pred_change_6','exp91_pred_vol_3','exp91_pred_vol_6','exp91_pred_accel','exp91_pred_max_jump_6','duration_ratio','cost_escalation_percentage'];corr,meta=fit_residual_booster(oof,score,features,9101);meta['trajectory_is_causal']=True;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
