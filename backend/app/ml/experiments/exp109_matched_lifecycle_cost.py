"""Exp109: production snapshot model plus lifecycle-matched project model."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import _family,_fit_pipeline,_mae,_regressors,PRODUCTION_COST_SEED,current_cost_oof,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_109';EXPERIMENT_NAME='Snapshot plus matched-lifecycle project Cost model';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=109
ALPHAS=(0.0,.25,.5,.75)
def _matched(frame):
    x=frame.copy();dr=pd.to_numeric(x['duration_ratio'],errors='coerce');x['_stage']=pd.cut(dr,[-np.inf,.2,.4,.7,1.0,np.inf],labels=False).fillna(-1).astype(int);x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x=x.sort_values(['canonical_project_id','_stage','snapshot_date']).drop_duplicates(['canonical_project_id','_stage'],keep='last').copy();counts=x.groupby('canonical_project_id')['canonical_project_id'].transform('size').clip(lower=1);x['sample_weight']=1.0/counts.astype(float);return x.drop(columns=['_stage'])
def _fit(train,score,features,family):
    t=_matched(train);m=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],t,features,'actual_cost_overrun_percentage');return np.asarray(m.predict(score.reindex(columns=features)),float),int(t['canonical_project_id'].nunique()),len(t)
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);features=list(ctx['cost_model'].features);family=_family(ctx['cost_model']);yy=pd.to_numeric(oof['oof_year'],errors='coerce');oof['exp109_project_model']=np.nan
    for year in sorted(int(v) for v in yy.dropna().unique()):
        cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[cy<year];val=oof.loc[yy==year]
        if fit.empty or val.empty:continue
        p,_,_=_fit(fit,val,features,family);oof.loc[yy==year,'exp109_project_model']=p
    valid=oof.dropna(subset=['exp109_project_model']);scores=[]
    for a in ALPHAS:
        p=(1-a)*valid['production_prediction'].to_numpy(float)+a*valid['exp109_project_model'].to_numpy(float);scores.append((_mae(valid['actual_cost_overrun_percentage'],p,valid['sample_weight']),a))
    _,alpha=min(scores,key=lambda z:(z[0],z[1]));candidate,nproj,nrows=_fit(ctx['train'],ctx['cohort'],features,family);pred=(1-alpha)*ctx['production_cost']+alpha*candidate;details={'lifecycle_bins':[.2,.4,.7,1.0],'sampling':'last training snapshot per project per lifecycle bin','matched_training_projects':nproj,'matched_training_rows':nrows,'blend_grid':list(ALPHAS),'selected_project_model_weight':alpha,'oof_scores':[{'mae':s,'weight':a} for s,a in scores]};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
