"""Experiment 83: project-bootstrap median Cost residual calibration."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.post_u1_cost_common import _mae,_wq,current_cost_oof,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_83';EXPERIMENT_NAME='Bagged median Cost residual calibration';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=83
FEATURES=['production_prediction','cost_escalation_percentage','expenditure_ratio','duration_ratio','schedule_slippage_days','approved_cost_cr','elapsed_days','planned_duration_days']

def _bag_predict(train,score,n_bags,seed):
    ids=train['canonical_project_id'].astype('string').dropna().unique();rng=np.random.default_rng(seed);pred=[]
    for b in range(n_bags):
        sampled=rng.choice(ids,size=len(ids),replace=True);bag=pd.concat([train.loc[train['canonical_project_id'].astype('string').eq(pid)] for pid in sampled],ignore_index=True)
        _,_,xb,xs=numeric_design(bag,score,FEATURES);m=LGBMRegressor(n_estimators=90,learning_rate=.025,max_depth=3,num_leaves=7,min_child_samples=70,reg_alpha=6,reg_lambda=24,random_state=seed+b,verbosity=-1,n_jobs=1);r=pd.to_numeric(bag['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(bag['sample_weight'],errors='coerce').fillna(0).to_numpy(float);m.fit(xb,r,sample_weight=w);pred.append(np.asarray(m.predict(xs),float))
    correction=np.median(np.vstack(pred),axis=0);rall=pd.to_numeric(train['residual'],errors='coerce').fillna(0).to_numpy(float);wall=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);cap=max(_wq(np.abs(rall),wall,.9),1e-9);return np.clip(correction,-cap,cap)

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);years=sorted(int(x) for x in pd.to_numeric(oof['oof_year'],errors='coerce').dropna().unique());meta=[]
    for year in years[1:]:
        yy=pd.to_numeric(oof['oof_year'],errors='coerce');fit=oof.loc[yy<year].copy();val=oof.loc[yy==year].copy()
        if len(fit)<80 or val.empty: continue
        meta.append((val,_bag_predict(fit,val,5,8300+year)))
    best=(float('inf'),0.0)
    for scale in (0.0,.25,.5,.75,1.0):
        vals=[];weights=[]
        for val,c in meta:
            p=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float)+scale*c;y=pd.to_numeric(val['actual_cost_overrun_percentage'],errors='coerce').to_numpy(float);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);vals.append(_mae(y,p,w));weights.append(max(float(w.sum()),1e-9))
        if vals: best=min(best,(float(np.average(vals,weights=weights)),scale))
    scale=best[1];score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];corr=_bag_predict(oof,score,9,8383);pred=ctx['production_cost']+scale*corr;details={'selected_scale':scale,'final_bags':9,'meta_bags':5,'aggregation':'median','bootstrap_unit':'project','features':FEATURES,'meta_oof_years':years[1:]};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
