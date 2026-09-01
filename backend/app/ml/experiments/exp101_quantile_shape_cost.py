"""Exp101: Cost quantile-shape uncertainty features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_101';EXPERIMENT_NAME='Quantile-shape Cost correction';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=101
QS=(.25,.5,.75)
def _fit_quantiles(train,score,features,seed):
    _,_,xt,xs=numeric_design(train,score,features);y=pd.to_numeric(train['actual_cost_overrun_percentage'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);out=[]
    for i,q in enumerate(QS):
        m=LGBMRegressor(objective='quantile',alpha=q,n_estimators=180,learning_rate=.03,max_depth=4,num_leaves=12,min_child_samples=60,reg_lambda=15,reg_alpha=3,random_state=seed+i,verbosity=-1,n_jobs=1);m.fit(xt,y,sample_weight=w);out.append(np.asarray(m.predict(xs),float))
    return out
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);features=list(ctx['cost_model'].features);yy=pd.to_numeric(oof['oof_year'],errors='coerce')
    for q in QS:oof[f'exp101_q{int(q*100)}']=np.nan
    for year in sorted(int(v) for v in yy.dropna().unique()):
        cy=pd.to_numeric(ctx['train']['completion_year'],errors='coerce');fit=ctx['train'].loc[cy<year];val=oof.loc[yy==year]
        if fit.empty or val.empty:continue
        for q,p in zip(QS,_fit_quantiles(fit,val,features,10100+year)):oof.loc[yy==year,f'exp101_q{int(q*100)}']=p
    score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];preds=_fit_quantiles(ctx['train'],score,features,10101)
    for q,p in zip(QS,preds):score[f'exp101_q{int(q*100)}']=p
    for f in (oof,score):
        f['exp101_interval_width']=f['exp101_q75']-f['exp101_q25'];f['exp101_upper_asymmetry']=f['exp101_q75']-f['exp101_q50'];f['exp101_lower_asymmetry']=f['exp101_q50']-f['exp101_q25']
    meta_features=['production_prediction','exp101_q25','exp101_q50','exp101_q75','exp101_interval_width','exp101_upper_asymmetry','exp101_lower_asymmetry','duration_ratio','cost_escalation_percentage'];corr,meta=fit_residual_booster(oof,score,meta_features,10102);meta['quantiles']=list(QS);return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
