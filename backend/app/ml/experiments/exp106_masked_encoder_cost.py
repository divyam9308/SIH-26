"""Exp106: self-supervised masked tabular trajectory-state encoder for Cost."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_106';EXPERIMENT_NAME='Self-supervised masked execution-state encoder';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=106
BASE=['cost_escalation_percentage','expenditure_ratio','physical_progress','schedule_slippage_days','duration_ratio','progress_deviation','approved_cost_cr','revised_cost_cr','elapsed_duration_days']
HIDDEN=12

def _encode(train,score,seed):
    _,_,xt,xs=numeric_design(train,score,BASE);sc=StandardScaler();zt=sc.fit_transform(xt);zs=sc.transform(xs);rng=np.random.default_rng(seed);corrupt=zt.copy();mask=rng.random(corrupt.shape)<.18;corrupt[mask]=0.0;m=MLPRegressor(hidden_layer_sizes=(HIDDEN,),activation='tanh',solver='adam',alpha=.05,learning_rate_init=.002,max_iter=140,random_state=seed);m.fit(corrupt,zt);return np.tanh(zs@m.coefs_[0]+m.intercepts_[0])
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);yy=pd.to_numeric(oof['oof_year'],errors='coerce');parts=[]
    for year in sorted(int(v) for v in yy.dropna().unique())[1:]:
        fit=oof.loc[yy<year];val=oof.loc[yy==year].copy()
        if len(fit)<100 or val.empty:continue
        h=_encode(fit,val,10600+year)
        for j in range(HIDDEN):val[f'exp106_z{j}']=h[:,j]
        parts.append(val)
    if not parts:raise ValueError('No forward self-supervised encoder folds')
    meta=pd.concat(parts,ignore_index=True);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];h=_encode(oof,score,10601)
    for j in range(HIDDEN):score[f'exp106_z{j}']=h[:,j]
    features=['production_prediction']+[f'exp106_z{j}' for j in range(HIDDEN)]+['duration_ratio','cost_escalation_percentage'];corr,details=fit_residual_booster(meta,score,features,10602);details.update({'encoder_target':'reconstruct standardized as-of covariates only','mask_probability':.18,'hidden_units':HIDDEN,'cost_target_used_in_encoder':False});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
