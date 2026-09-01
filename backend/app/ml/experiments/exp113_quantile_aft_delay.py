"""Experiment 113: quantile AFT uncertainty features beyond current U1 Delay."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.exp35_aft_residual_combo import _delay_from_remaining
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,numeric_design,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_113';EXPERIMENT_NAME='Quantile AFT uncertainty stack';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=113
QUANTILES=(.25,.5,.75)
BASE_FEATURES=['production_prediction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','progress_deviation','approved_cost_cr','planned_duration_days','elapsed_duration_days','exp58_delay_hier_prior','exp58_group_support']

def _remaining(frame): return (pd.to_datetime(frame['completion_date'],errors='coerce')-pd.to_datetime(frame['snapshot_date'],errors='coerce')).dt.days.clip(lower=1).astype(float)
def _quantiles(train,score,seed):
    _,_,xt,xs=numeric_design(train,score,BASE_FEATURES);y=np.log1p(_remaining(train).to_numpy(float));w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);out={}
    for i,q in enumerate(QUANTILES):
        m=LGBMRegressor(objective='quantile',alpha=q,n_estimators=160,learning_rate=.025,max_depth=3,num_leaves=10,min_child_samples=70,subsample=.85,colsample_bytree=.85,reg_alpha=5,reg_lambda=25,random_state=seed+i,verbosity=-1,n_jobs=1);m.fit(xt,y,sample_weight=w);rem=np.maximum(1,np.expm1(np.clip(np.asarray(m.predict(xs),float),-10,10)));out[q]=rem
    return out

def _attach(train,score,seed):
    q=_quantiles(train,score,seed);p=score.copy();d25=_delay_from_remaining(score,q[.25]);d50=_delay_from_remaining(score,q[.5]);d75=_delay_from_remaining(score,q[.75]);p['exp113_q50_delay']=d50;p['exp113_interval_width']=q[.75]-q[.25];p['exp113_upper_asymmetry']=q[.75]-q[.5];p['exp113_lower_asymmetry']=q[.5]-q[.25];return p

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);ys=pd.to_numeric(oof['oof_year'],errors='coerce');years=sorted(int(x) for x in ys.dropna().unique());parts=[]
    for year in years[1:]:
        fit=oof.loc[ys<year].copy();val=oof.loc[ys==year].copy()
        if len(fit)<120 or val.empty: continue
        parts.append(_attach(fit,val,11300+year))
    if not parts: raise ValueError('No forward quantile-AFT evidence')
    meta=pd.concat(parts,ignore_index=True);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;score=_attach(oof,score,11400+training_end)
    features=['production_prediction','u1_correction','exp113_q50_delay','exp113_interval_width','exp113_upper_asymmetry','exp113_lower_asymmetry','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','exp58_group_support'];corr,details=fit_residual_booster(meta,score,features,11301);details.update({'quantiles':list(QUANTILES),'quantile_meta_predictions_are_forward_oof':True});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
