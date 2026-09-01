"""Exp95: time-varying hierarchical Cost residual priors."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_95';EXPERIMENT_NAME='Time-varying hierarchical Cost residual priors';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=95
HORIZONS=(3,5,8)
def _weighted_group(history,key):
    h=history.copy();h['_wr']=pd.to_numeric(h['residual'],errors='coerce')*pd.to_numeric(h['sample_weight'],errors='coerce').fillna(0);g=h.groupby(key,dropna=False).agg(_sum=('_wr','sum'),_w=('sample_weight','sum'),_n=('canonical_project_id','nunique'));return (g['_sum']/g['_w'].replace(0,np.nan)).where(g['_n']>=3)
def _add(oof,score,training_end):
    yy=pd.to_numeric(oof['oof_year'],errors='coerce')
    for horizon in HORIZONS:
        for key in ('implementing_agency','sector'):
            col=f'exp95_{key}_{horizon}y';oof[col]=np.nan
            for year in sorted(int(v) for v in yy.dropna().unique()):
                hist=oof.loc[(yy<year)&(yy>=year-horizon)].copy();val=oof.loc[yy==year]
                if hist.empty or val.empty: continue
                mp=_weighted_group(hist,key);oof.loc[yy==year,col]=val[key].map(mp).to_numpy()
            hist=oof.loc[yy>=training_end-horizon+1].copy();score[col]=score[key].map(_weighted_group(hist,key)) if not hist.empty else np.nan
    return oof,score
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];oof,score=_add(oof,score,training_end);features=['production_prediction']+[f'exp95_{k}_{h}y' for h in HORIZONS for k in ('implementing_agency','sector')]+['exp58_group_support','duration_ratio','cost_escalation_percentage'];corr,meta=fit_residual_booster(oof,score,features,9501);meta['prior_horizons_years']=list(HORIZONS);meta['minimum_group_projects']=3;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
