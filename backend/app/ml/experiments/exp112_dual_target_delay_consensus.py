"""Experiment 112: dual-target completion consensus beyond current U1 Delay."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.exp35_aft_residual_combo import _delay_from_remaining
from backend.app.ml.experiments.post_u1_delay_common import _mae,current_delay_oof,numeric_design,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_112';EXPERIMENT_NAME='Dual-target completion consensus';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=112
FEATURES=['production_prediction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','progress_deviation','approved_cost_cr','planned_duration_days','elapsed_duration_days','exp58_delay_hier_prior','exp58_group_support']
GRID=(0.0,.25,.5,.75,1.0)

def _remaining(frame): return (pd.to_datetime(frame['completion_date'],errors='coerce')-pd.to_datetime(frame['snapshot_date'],errors='coerce')).dt.days.clip(lower=1).astype(float)

def _fit_model(train,score,target,seed):
    _,_,xt,xs=numeric_design(train,score,FEATURES);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float)
    m=LGBMRegressor(n_estimators=160,learning_rate=.025,max_depth=3,num_leaves=10,min_child_samples=70,subsample=.85,colsample_bytree=.85,reg_alpha=5,reg_lambda=25,random_state=seed,verbosity=-1,n_jobs=1);m.fit(xt,np.asarray(target,float),sample_weight=w);return np.asarray(m.predict(xs),float)

def _aux_predictions(train,score,seed):
    rem=np.log1p(_remaining(train).to_numpy(float));pred_rem=np.maximum(1,np.expm1(np.clip(_fit_model(train,score,rem,seed),-10,10)));a=_delay_from_remaining(score,pred_rem)
    sl=pd.to_numeric(train.get('schedule_slippage_days'),errors='coerce').fillna(0).to_numpy(float);actual=pd.to_numeric(train['actual_delay_days'],errors='coerce').to_numpy(float);future_error=actual-sl
    err=_fit_model(train,score,future_error,seed+1);score_sl=pd.to_numeric(score.get('schedule_slippage_days'),errors='coerce').fillna(0).to_numpy(float);b=np.maximum(0,score_sl+err)
    return a,b

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);ys=pd.to_numeric(oof['oof_year'],errors='coerce');years=sorted(int(x) for x in ys.dropna().unique());meta=[]
    for year in years[1:]:
        fit=oof.loc[ys<year].copy();val=oof.loc[ys==year].copy()
        if len(fit)<120 or val.empty: continue
        a,b=_aux_predictions(fit,val,11200+year);meta.append((val,a,b))
    if not meta: raise ValueError('No forward dual-target predictions')
    best=(float('inf'),0.0,0.0)
    for wa in GRID:
        for wb in GRID:
            if wa+wb>1.000001: continue
            vals=[];weights=[]
            for val,a,b in meta:
                prod=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float);aa=np.where(np.isfinite(a),a,prod);bb=np.where(np.isfinite(b),b,prod);p=np.maximum(0,(1-wa-wb)*prod+wa*aa+wb*bb);y=pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);vals.append(_mae(y,p,w));weights.append(max(float(np.nansum(w)),1e-9))
            best=min(best,(float(np.average(vals,weights=weights)),wa,wb))
    wa,wb=float(best[1]),float(best[2]);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];a,b=_aux_predictions(oof,score,11300+training_end);prod=ctx['production_delay'];a=np.where(np.isfinite(a),a,prod);b=np.where(np.isfinite(b),b,prod);final=np.maximum(0,(1-wa-wb)*prod+wa*a+wb*b)
    return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,final,{'production_weight':1-wa-wb,'remaining_time_weight':wa,'schedule_error_weight':wb,'selection_years':years[1:],'targets':['log1p_remaining_days','signed_future_schedule_error'],'features':FEATURES},output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
