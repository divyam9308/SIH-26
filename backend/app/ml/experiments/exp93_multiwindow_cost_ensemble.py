"""Exp93: OOF-selected ensemble of independently fitted trailing-history Cost models."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import _family,_fit_pipeline,_mae,_regressors,PRODUCTION_COST_SEED,current_cost_oof,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_93';EXPERIMENT_NAME='Multi-training-window Cost ensemble';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=93
WINDOWS=(5,7,10,15);BLENDS=(0.0,.25,.5,.75,1.0)
def _fit_predict(fit,score,features,family):
    m=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],fit,features,'actual_cost_overrun_percentage');return np.asarray(m.predict(score.reindex(columns=features)),float)
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);features=list(ctx['cost_model'].features);family=_family(ctx['cost_model']);yy=pd.to_numeric(oof['oof_year'],errors='coerce');scores=[]
    for years in WINDOWS:
        col=f'exp93_w{years}';oof[col]=np.nan
        for year in sorted(int(v) for v in yy.dropna().unique()):
            val=oof.loc[yy==year];cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[(cy<year)&(cy>=year-years)].copy()
            if fit['canonical_project_id'].nunique()<20 or val.empty: continue
            oof.loc[yy==year,col]=_fit_predict(fit,val,features,family)
        valid=oof[[col,'production_prediction','actual_cost_overrun_percentage','sample_weight']].dropna()
        for alpha in BLENDS:
            pred=(1-alpha)*valid['production_prediction'].to_numpy(float)+alpha*valid[col].to_numpy(float);scores.append((_mae(valid['actual_cost_overrun_percentage'],pred,valid['sample_weight']),years,alpha))
    best=min(scores,key=lambda t:(t[0],t[2],-t[1]));_,years,alpha=best;cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[cy>=training_end-years+1].copy();candidate=_fit_predict(fit,ctx['cohort'],features,family);pred=(1-alpha)*ctx['production_cost']+alpha*candidate;details={'candidate_windows_years':list(WINDOWS),'blend_grid':list(BLENDS),'selected_window_years':years,'selected_trailing_weight':alpha,'oof_scores':[{'mae':s,'years':y,'weight':a} for s,y,a in scores]};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
