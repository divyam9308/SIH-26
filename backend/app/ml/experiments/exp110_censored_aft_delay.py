"""Experiment 110: gradient-boosted censored-prefix AFT signal beyond U1.

The supervised repository contract contains completed training projects only, so
this experiment does not leak holdout projects in as "active" training rows.
Instead it creates a training-only right-censored view: if a historical OOF row
was still more than 365 days from completion, its AFT label is recorded only as
>365 days. Rows completing within that horizon retain exact remaining time.
A boosted XGBoost AFT model learns from those interval labels and is blended with
current production using forward OOF evidence only.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from backend.app.ml.experiments.exp35_aft_residual_combo import _delay_from_remaining
from backend.app.ml.experiments.post_u1_delay_common import (
    _mae,current_delay_oof,numeric_design,persist,prepare_context,window_contract,
)

EXPERIMENT_ID='exp_110'
EXPERIMENT_NAME='Gradient-boosted censored-prefix AFT Delay signal'
EXPERIMENT_SCOPE='delay'
EXPERIMENT_SEQUENCE=110
CENSOR_DAYS=365.0
FEATURES=['production_prediction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','progress_deviation','approved_cost_cr','planned_duration_days','exp58_delay_hier_prior','exp58_group_support']


def _remaining(frame):
    return (pd.to_datetime(frame['completion_date'],errors='coerce')-pd.to_datetime(frame['snapshot_date'],errors='coerce')).dt.days.clip(lower=1).astype(float)


def _aft_predict(train,score,seed):
    _,_,xt,xs=numeric_design(train,score,FEATURES)
    rem=_remaining(train).to_numpy(float)
    lower=np.minimum(rem,CENSOR_DAYS)
    upper=np.where(rem<=CENSOR_DAYS,rem,np.inf)
    d=xgb.DMatrix(xt.to_numpy(float),feature_names=list(xt.columns))
    d.set_float_info('label_lower_bound',lower)
    d.set_float_info('label_upper_bound',upper)
    w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float)
    d.set_weight(w)
    params={'objective':'survival:aft','eval_metric':'aft-nloglik','aft_loss_distribution':'normal','aft_loss_distribution_scale':1.0,'max_depth':3,'eta':0.03,'subsample':0.85,'colsample_bytree':0.85,'lambda':20.0,'alpha':4.0,'seed':seed,'tree_method':'hist','nthread':1}
    model=xgb.train(params,d,num_boost_round=140,verbose_eval=False)
    ds=xgb.DMatrix(xs.to_numpy(float),feature_names=list(xs.columns))
    return np.clip(np.asarray(model.predict(ds),float),1.0,6000.0)


def fit_experiment(training_end,output):
    window_contract(training_end)
    ctx=prepare_context(training_end)
    oof=current_delay_oof(ctx['train'],ctx['delay_model'])
    ys=pd.to_numeric(oof['oof_year'],errors='coerce')
    years=sorted(int(x) for x in ys.dropna().unique())
    meta=[]
    for year in years[1:]:
        fit=oof.loc[ys<year].copy(); val=oof.loc[ys==year].copy()
        if len(fit)<120 or val.empty: continue
        rem=_aft_predict(fit,val,11000+year)
        cand=_delay_from_remaining(val,rem)
        meta.append((val,cand))
    if not meta: raise ValueError('No forward censored-AFT predictions')
    best=(float('inf'),0.0)
    for w_cand in (0.0,.25,.5,.75,1.0):
        vals=[];weights=[]
        for val,cand in meta:
            prod=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float)
            pred=np.maximum(0,(1-w_cand)*prod+w_cand*cand)
            y=pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float)
            w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float)
            vals.append(_mae(y,pred,w));weights.append(max(float(np.nansum(w)),1e-9))
        score=float(np.average(vals,weights=weights))
        best=min(best,(score,w_cand))
    selected=float(best[1])
    score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay']
    rem=_aft_predict(oof,score,11110+training_end)
    cand=_delay_from_remaining(score,rem)
    final=np.maximum(0,(1-selected)*ctx['production_delay']+selected*cand)
    details={'censor_horizon_days':CENSOR_DAYS,'selected_candidate_weight':selected,'selection_years':years[1:],'active_holdout_projects_used_for_training':False,'censoring_semantics':'training-only pseudo-censored prefixes; >365d encoded [365,+inf)','features':FEATURES}
    return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,final,details,output)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
