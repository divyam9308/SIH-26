"""Exp94: tree-family disagreement as Cost uncertainty signal."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import _fit_pipeline,_regressors,PRODUCTION_COST_SEED,current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_94';EXPERIMENT_NAME='Cost model-family disagreement booster';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=94
FAMILIES=('extra_trees','lightgbm','xgboost')
def _add_oof(ctx,oof):
    features=list(ctx['cost_model'].features);yy=pd.to_numeric(oof['oof_year'],errors='coerce')
    for fam in FAMILIES: oof[f'exp94_{fam}']=np.nan
    for year in sorted(int(v) for v in yy.dropna().unique()):
        cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[cy<year].copy();val=oof.loc[yy==year]
        if fit.empty or val.empty: continue
        for fam in FAMILIES:
            m=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],fit,features,'actual_cost_overrun_percentage');oof.loc[yy==year,f'exp94_{fam}']=np.asarray(m.predict(val.reindex(columns=features)),float)
    arr=oof[[f'exp94_{f}' for f in FAMILIES]].to_numpy(float);oof['exp94_family_std']=np.nanstd(arr,axis=1);oof['exp94_family_range']=np.nanmax(arr,axis=1)-np.nanmin(arr,axis=1);return oof
def _add_score(ctx,score):
    features=list(ctx['cost_model'].features);vals=[]
    for fam in FAMILIES:
        m=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],ctx['train'],features,'actual_cost_overrun_percentage');p=np.asarray(m.predict(score.reindex(columns=features)),float);score[f'exp94_{fam}']=p;vals.append(p)
    arr=np.vstack(vals).T;score['exp94_family_std']=np.std(arr,axis=1);score['exp94_family_range']=np.ptp(arr,axis=1);return score
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=_add_oof(ctx,current_cost_oof(ctx['train'],ctx['cost_model']));score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];score=_add_score(ctx,score);features=['production_prediction','exp94_family_std','exp94_family_range']+[f'exp94_{f}' for f in FAMILIES]+['duration_ratio','cost_escalation_percentage'];corr,meta=fit_residual_booster(oof,score,features,9401);meta['families']=list(FAMILIES);return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
