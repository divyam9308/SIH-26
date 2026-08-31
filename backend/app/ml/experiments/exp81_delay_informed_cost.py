"""Experiment 81: Delay-informed Cost residual calibration."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
from backend.app.ml.production_u1_delay_baseline import _delay_oof_frame,_fit_u1_booster
EXPERIMENT_ID='exp_81';EXPERIMENT_NAME='Delay-informed Cost residual calibration';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=81

def current_delay_meta_oof(train,delay_model):
    if not hasattr(delay_model,'base_model'): raise TypeError('U1 production Delay wrapper required')
    base=_delay_oof_frame(train,delay_model.base_model);ys=pd.to_numeric(base['oof_year'],errors='coerce');years=sorted(int(x) for x in ys.dropna().unique());parts=[]
    for year in years[1:]:
        fit=base.loc[ys<year].copy();val=base.loc[ys==year].copy()
        if len(fit)<80 or val.empty: continue
        _,_,_,_,corr=_fit_u1_booster(fit,val);anchor=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float);pred=np.maximum(0,anchor+corr);p=val[['canonical_project_id','snapshot_date']].copy();p['delay_oof_prediction']=pred;p['u1_delay_correction']=pred-anchor;p['oof_year']=year;parts.append(p)
    if not parts: raise ValueError('No Delay meta-OOF rows')
    return pd.concat(parts,ignore_index=True)

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);cost=current_cost_oof(ctx['train'],ctx['cost_model']);delay=current_delay_meta_oof(ctx['train'],ctx['delay_model']);oof=cost.merge(delay[['canonical_project_id','snapshot_date','delay_oof_prediction','u1_delay_correction']],on=['canonical_project_id','snapshot_date'],how='inner')
    score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];score['delay_oof_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_delay_correction']=ctx['production_delay']-base
    features=['production_prediction','delay_oof_prediction','u1_delay_correction','cost_escalation_percentage','expenditure_ratio','duration_ratio','schedule_slippage_days','approved_cost_cr'];corr,meta=fit_residual_booster(oof,score,features,8101);meta['delay_signal_source']='forward OOF current-production Delay only';return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
