"""Exp105: forward-fitted multivariate execution-factor Cost features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_105';EXPERIMENT_NAME='Dynamic multivariate execution-factor Cost model';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=105
BASE=['cost_escalation_percentage','expenditure_ratio','progress_deviation','schedule_slippage_days','duration_ratio','physical_progress']
def _embed(train,score,seed):
    _,_,xt,xs=numeric_design(train,score,BASE);sc=StandardScaler();zt=sc.fit_transform(xt);zs=sc.transform(xs);fa=FactorAnalysis(n_components=2,random_state=seed,max_iter=500);fa.fit(zt);return fa.transform(zs)
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);yy=pd.to_numeric(oof['oof_year'],errors='coerce');parts=[]
    for year in sorted(int(v) for v in yy.dropna().unique())[1:]:
        fit=oof.loc[yy<year];val=oof.loc[yy==year].copy()
        if len(fit)<80 or val.empty:continue
        z=_embed(fit,val,10500+year);val['exp105_factor_1']=z[:,0];val['exp105_factor_2']=z[:,1];parts.append(val)
    if not parts:raise ValueError('No forward factor folds')
    meta=pd.concat(parts,ignore_index=True);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];z=_embed(oof,score,10501);score['exp105_factor_1']=z[:,0];score['exp105_factor_2']=z[:,1];features=['production_prediction','exp105_factor_1','exp105_factor_2','cost_escalation_percentage','duration_ratio'];corr,details=fit_residual_booster(meta,score,features,10502);details['factor_inputs']=BASE;details['factor_fit_for_meta']='strictly earlier OOF years';return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
