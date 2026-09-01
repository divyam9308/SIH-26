"""Exp108: OOF hard-project class reweighting for Cost."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import _family,_fit_pipeline,_mae,_regressors,PRODUCTION_COST_SEED,current_cost_oof,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_108';EXPERIMENT_NAME='OOF hard-project Cost reweighting';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=108
ALPHAS=(0.0,.25,.5,.75)
def _key(frame):
    x=frame.copy();dr=pd.to_numeric(x['duration_ratio'],errors='coerce');stage=pd.cut(dr,[-np.inf,.3,.7,1.1,np.inf],labels=['early','mid','late','very_late']).astype('string').fillna('unknown');return x['sector'].astype('string').fillna('unknown')+'|'+x['project_size_category'].astype('string').fillna('unknown')+'|'+stage
def _difficulty(hist):
    h=hist.copy();h['_key']=_key(h);h['_abs']=pd.to_numeric(h['residual'],errors='coerce').abs();g=h.groupby('_key').agg(err=('_abs','mean'),n=('canonical_project_id','nunique'));base=float(h['_abs'].mean()) if len(h) else 1.0;factor=(g.err/max(base,1e-9)).clip(.75,1.5);return factor.where(g.n>=3,1.0)
def _weighted_fit(train,dmap,features,family):
    t=train.copy();fac=_key(t).map(dmap).fillna(1.0).to_numpy(float);w=pd.to_numeric(t['sample_weight'],errors='coerce').fillna(0).to_numpy(float);nw=w*fac
    if nw.sum()>0:nw*=w.sum()/nw.sum()
    t['sample_weight']=nw;return _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],t,features,'actual_cost_overrun_percentage')
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);features=list(ctx['cost_model'].features);family=_family(ctx['cost_model']);yy=pd.to_numeric(oof['oof_year'],errors='coerce');oof['exp108_candidate']=np.nan
    for year in sorted(int(v) for v in yy.dropna().unique())[1:]:
        hist=oof.loc[yy<year];val=oof.loc[yy==year];cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[cy<year]
        if hist.empty or val.empty or fit.empty:continue
        m=_weighted_fit(fit,_difficulty(hist),features,family);oof.loc[yy==year,'exp108_candidate']=np.asarray(m.predict(val.reindex(columns=features)),float)
    valid=oof.dropna(subset=['exp108_candidate']);scores=[]
    for a in ALPHAS:
        p=(1-a)*valid['production_prediction'].to_numpy(float)+a*valid['exp108_candidate'].to_numpy(float);scores.append((_mae(valid['actual_cost_overrun_percentage'],p,valid['sample_weight']),a))
    _,alpha=min(scores,key=lambda z:(z[0],z[1]));m=_weighted_fit(ctx['train'],_difficulty(oof),features,family);candidate=np.asarray(m.predict(ctx['cohort'].reindex(columns=features)),float);pred=(1-alpha)*ctx['production_cost']+alpha*candidate;details={'class_key':'sector|project_size|lifecycle_stage','difficulty_clip':[.75,1.5],'minimum_projects':3,'blend_grid':list(ALPHAS),'selected_reweighted_model_weight':alpha,'oof_scores':[{'mae':s,'weight':a} for s,a in scores]};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
