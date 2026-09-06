"""Tail-weighted remaining-time expert with cross-fitted benefit routing over Exp113."""
from __future__ import annotations
import argparse
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor
from backend.app.ml.experiments.post_exp113_component_common import build_component_oof_fold,component_context,component_oof,load_component_oof_dir
from backend.app.ml.experiments.post_exp113_delay_common import numeric_design,persist
EXP_ID='exp_delay_remaining_tail_benefit_gate';NAME='Remaining-Time Tail Expert with Benefit Routing';ALPHAS=(.50,.60,.70);TAIL_WEIGHTS=(1.,2.,4.);SCALES=(.25,.5,.75,1.)
FEATURES=['production_prediction','production_u1','exp113_correction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','approved_cost_cr','planned_duration_days','elapsed_duration_days','exp58_delay_hier_prior','exp58_group_support']
def _remaining_target(frame):
    actual=pd.to_datetime(frame.get('actual_completion_date'),errors='coerce');snapshot=pd.to_datetime(frame.get('snapshot_date'),errors='coerce');days=(actual-snapshot).dt.days;fallback=pd.to_numeric(frame['actual_delay_days'],errors='coerce').clip(lower=0);return np.log1p(days.where(days.notna() & (days>=0),fallback).fillna(0).to_numpy(float))
def _to_total_delay(frame,log_remaining):
    snapshot=pd.to_datetime(frame.get('snapshot_date'),errors='coerce');planned=pd.to_datetime(frame.get('planned_completion_date'),errors='coerce');remaining=np.maximum(np.expm1(np.asarray(log_remaining,float)),0);elapsed=(snapshot-planned).dt.days.to_numpy(float);return np.maximum(0,np.nan_to_num(elapsed,nan=0)+remaining)
def _fit_tail(fit,score,alpha,multiplier,seed):
    _,_,xf,xs=numeric_design(fit,score,FEATURES);target=_remaining_target(fit);w=pd.to_numeric(fit['sample_weight'],errors='coerce').fillna(0).to_numpy(float);actual=pd.to_numeric(fit['actual_delay_days'],errors='coerce').fillna(0).to_numpy(float);prod=pd.to_numeric(fit['production_prediction'],errors='coerce').fillna(0).to_numpy(float);q90=float(np.nanquantile(actual,.90));tail=(actual>=q90)&((actual-prod)>0);w=w*np.where(tail,multiplier,1.);m=LGBMRegressor(objective='quantile',alpha=alpha,n_estimators=180,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=50,reg_alpha=5,reg_lambda=30,random_state=seed,verbosity=-1,n_jobs=1);m.fit(xf,target,sample_weight=w);return _to_total_delay(score,m.predict(xs))
def _crossfit_tail(oof,alpha,multiplier):
    years=sorted(int(v) for v in pd.to_numeric(oof['oof_year'],errors='coerce').dropna().unique());pred=pd.Series(np.nan,index=oof.index,dtype=float)
    for year in years[1:]:
        yc=pd.to_numeric(oof['oof_year'],errors='coerce');fit=oof[yc<year];val=oof[yc==year]
        if len(fit)>=100 and not val.empty:pred.loc[val.index]=_fit_tail(fit,val,alpha,multiplier,11350+year)
    return pred
def _crossfit_gate(work):
    years=sorted(int(v) for v in pd.to_numeric(work['oof_year'],errors='coerce').dropna().unique());gate=pd.Series(0.,index=work.index,dtype=float)
    for year in years[1:]:
        yc=pd.to_numeric(work['oof_year'],errors='coerce');fit=work[yc<year];val=work[yc==year]
        if len(fit)<100 or val.empty or fit['benefit'].nunique()<2:continue
        _,_,xf,xv=numeric_design(fit,val,FEATURES+['tail_prediction','tail_uplift']);m=LGBMClassifier(n_estimators=120,learning_rate=.03,max_depth=2,num_leaves=4,min_child_samples=50,reg_alpha=5,reg_lambda=30,class_weight='balanced',random_state=11450+year,verbosity=-1,n_jobs=1);m.fit(xf,fit['benefit'].to_numpy(int),sample_weight=fit['sample_weight'].to_numpy(float));gate.loc[val.index]=m.predict_proba(xv)[:,1]
    return gate
def run(output,oof=None):
    ctx=component_context();oof=component_oof(ctx) if oof is None else oof;y=pd.to_numeric(oof['actual_delay_days'],errors='coerce').to_numpy(float);anchor=pd.to_numeric(oof['production_prediction'],errors='coerce').to_numpy(float);best=None;best_work=None
    for alpha in ALPHAS:
        for multiplier in TAIL_WEIGHTS:
            tail=_crossfit_tail(oof,alpha,multiplier);work=oof.copy();work['tail_prediction']=tail;work['tail_uplift']=np.maximum(tail-work['production_prediction'],0);valid=tail.notna();actual_v=y[valid];anchor_v=anchor[valid];tail_v=tail[valid].to_numpy(float);work.loc[valid,'benefit']=(np.abs(actual_v-tail_v)+30<np.abs(actual_v-anchor_v)).astype(int);vw=work.loc[valid].copy();gate=_crossfit_gate(vw);cap=float(np.nanquantile(np.maximum(pd.to_numeric(vw['residual'],errors='coerce'),0),.95))
            for scale in SCALES:
                idx=vw.index;corr=scale*gate.reindex(idx).fillna(0).to_numpy(float)*np.clip(vw['tail_uplift'].to_numpy(float),0,cap);pred=pd.to_numeric(vw['production_prediction'],errors='coerce').to_numpy(float)+corr;yy=pd.to_numeric(vw['actual_delay_days'],errors='coerce').to_numpy(float);ww=pd.to_numeric(vw['sample_weight'],errors='coerce').to_numpy(float);mae=float(np.average(np.abs(yy-pred),weights=ww));candidate=(mae,alpha,multiplier,scale,cap)
                if best is None or candidate[0]<best[0]:best=candidate;best_work=vw
    if best is None or best_work is None:raise ValueError('No valid nested forward tail-benefit configuration')
    _,alpha,multiplier,scale,cap=best;train=oof.copy();train['tail_prediction']=_fit_tail(oof,oof,alpha,multiplier,11501);train['tail_uplift']=np.maximum(train['tail_prediction']-train['production_prediction'],0);yy=pd.to_numeric(train['actual_delay_days'],errors='coerce').to_numpy(float);aa=pd.to_numeric(train['production_prediction'],errors='coerce').to_numpy(float);train['benefit']=(np.abs(yy-train['tail_prediction'].to_numpy(float))+30<np.abs(yy-aa)).astype(int);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];score['production_u1']=ctx['production_u1'];score['exp113_correction']=ctx['exp113_correction'];score['tail_prediction']=_fit_tail(oof,score,alpha,multiplier,11502);score['tail_uplift']=np.maximum(score['tail_prediction']-score['production_prediction'],0);_,_,xf,xs=numeric_design(train,score,FEATURES+['tail_prediction','tail_uplift']);gm=LGBMClassifier(n_estimators=140,learning_rate=.03,max_depth=2,num_leaves=4,min_child_samples=50,reg_alpha=5,reg_lambda=30,class_weight='balanced',random_state=11503,verbosity=-1,n_jobs=1);gm.fit(xf,train['benefit'].to_numpy(int),sample_weight=train['sample_weight'].to_numpy(float));gate=gm.predict_proba(xs)[:,1];prediction=ctx['production_delay']+scale*gate*np.clip(score['tail_uplift'].to_numpy(float),0,cap);details={'selected_alpha':alpha,'selected_tail_weight':multiplier,'selected_scale':scale,'correction_cap':cap,'nested_oof_mae':best[0],'holdout_used_for_selection':False,'full_holdout_retained':True,'oof_years':sorted(int(x) for x in pd.to_numeric(oof['oof_year']).unique())};return persist(EXP_ID,NAME,ctx,prediction,details,output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='test-output/exp-delay-remaining-tail-benefit-gate/result.json');p.add_argument('--oof-dir');p.add_argument('--build-oof-year',type=int);p.add_argument('--oof-output');a=p.parse_args()
    if a.build_oof_year is not None:
        if not a.oof_output:p.error('--oof-output is required with --build-oof-year')
        build_component_oof_fold(a.build_oof_year,a.oof_output)
    else:run(a.output,load_component_oof_dir(a.oof_dir) if a.oof_dir else None)
