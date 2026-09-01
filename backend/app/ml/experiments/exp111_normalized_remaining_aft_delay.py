"""Experiment 111: scale-normalized remaining-time AFT signal beyond U1."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.exp35_aft_residual_combo import _delay_from_remaining
from backend.app.ml.experiments.post_u1_delay_common import _mae,current_delay_oof,numeric_design,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_111';EXPERIMENT_NAME='Scale-normalized remaining-time AFT';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=111
FEATURES=['production_prediction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','progress_deviation','approved_cost_cr','planned_duration_days','elapsed_duration_days','exp58_delay_hier_prior','exp58_group_support']

def _remaining(frame): return (pd.to_datetime(frame['completion_date'],errors='coerce')-pd.to_datetime(frame['snapshot_date'],errors='coerce')).dt.days.clip(lower=1).astype(float)
def _duration(frame,median=365.0):
    s=pd.to_numeric(frame.get('planned_duration_days',pd.Series(np.nan,index=frame.index)),errors='coerce').replace([np.inf,-np.inf],np.nan)
    return s.where(s>1).fillna(float(median)).to_numpy(float)

def _fit_predict(train,score,seed):
    _,_,xt,xs=numeric_design(train,score,FEATURES)
    dur_train=_duration(train); med=float(np.nanmedian(dur_train)) if len(dur_train) else 365.0
    target=np.log1p(_remaining(train).to_numpy(float)/np.maximum(dur_train,1.0))
    w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float)
    model=LGBMRegressor(n_estimators=180,learning_rate=.025,max_depth=3,num_leaves=10,min_child_samples=70,subsample=.85,colsample_bytree=.85,reg_alpha=5,reg_lambda=25,random_state=seed,verbosity=-1,n_jobs=1)
    model.fit(xt,target,sample_weight=w)
    ratio=np.maximum(0,np.expm1(np.clip(np.asarray(model.predict(xs),float),-10,10)))
    return ratio*np.maximum(_duration(score,med),1.0)

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);ys=pd.to_numeric(oof['oof_year'],errors='coerce');years=sorted(int(x) for x in ys.dropna().unique());meta=[]
    for year in years[1:]:
        fit=oof.loc[ys<year].copy();val=oof.loc[ys==year].copy()
        if len(fit)<120 or val.empty: continue
        cand=_delay_from_remaining(val,_fit_predict(fit,val,11100+year));meta.append((val,cand))
    if not meta: raise ValueError('No normalized remaining-time forward predictions')
    best=(float('inf'),0.0)
    for wc in (0.0,.25,.5,.75,1.0):
        vals=[];weights=[]
        for val,cand in meta:
            prod=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float);p=np.maximum(0,(1-wc)*prod+wc*cand);y=pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);vals.append(_mae(y,p,w));weights.append(max(float(np.nansum(w)),1e-9))
        best=min(best,(float(np.average(vals,weights=weights)),wc))
    wc=float(best[1]);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];cand=_delay_from_remaining(score,_fit_predict(oof,score,11200+training_end));final=np.maximum(0,(1-wc)*ctx['production_delay']+wc*cand)
    return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,final,{'selected_normalized_aft_weight':wc,'selection_years':years[1:],'target':'log1p(remaining_days/planned_duration_days)','features':FEATURES},output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
