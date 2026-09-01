"""Exp102: distributionally robust Cost ensemble selected by worst OOF era."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import _fit_pipeline,_mae,_regressors,PRODUCTION_COST_SEED,current_cost_oof,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_102';EXPERIMENT_NAME='Distributionally robust Cost ensemble';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=102
FAMILIES=('extra_trees','lightgbm','xgboost');ALPHAS=(.25,.5,.75)
def _fit(family,train,score,features):
    m=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],train,features,'actual_cost_overrun_percentage');return np.asarray(m.predict(score.reindex(columns=features)),float)
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);features=list(ctx['cost_model'].features);yy=pd.to_numeric(oof['oof_year'],errors='coerce')
    for fam in FAMILIES:oof[f'exp102_{fam}']=np.nan
    for year in sorted(int(v) for v in yy.dropna().unique()):
        cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[cy<year];val=oof.loc[yy==year]
        if fit.empty or val.empty:continue
        for fam in FAMILIES:oof.loc[yy==year,f'exp102_{fam}']=_fit(fam,fit,val,features)
    candidates=[('production',None,0.0)] + [(fam,fam,a) for fam in FAMILIES for a in ALPHAS];results=[]
    years=sorted(int(v) for v in yy.dropna().unique())
    for name,fam,a in candidates:
        fold=[]
        for year in years:
            v=oof.loc[yy==year].dropna(subset=['production_prediction']+([] if fam is None else [f'exp102_{fam}']))
            if v.empty:continue
            p=v['production_prediction'].to_numpy(float) if fam is None else (1-a)*v['production_prediction'].to_numpy(float)+a*v[f'exp102_{fam}'].to_numpy(float);fold.append(_mae(v['actual_cost_overrun_percentage'],p,v['sample_weight']))
        if fold:results.append((max(fold),float(np.mean(fold)),name,fam,a,fold))
    best=min(results,key=lambda z:(z[0],z[1],z[4]));_,_,_,fam,a,_=best
    if fam is None:pred=ctx['production_cost']
    else:pred=(1-a)*ctx['production_cost']+a*_fit(fam,ctx['train'],ctx['cohort'],features)
    details={'criterion':'minimize worst forward-OOF year MAE, then mean MAE','selected_family':fam,'selected_family_weight':a,'candidates':[{'worst_mae':r[0],'mean_mae':r[1],'name':r[2],'family':r[3],'weight':r[4],'fold_mae':r[5]} for r in results]};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
