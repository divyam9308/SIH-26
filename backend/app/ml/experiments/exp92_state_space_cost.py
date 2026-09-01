"""Exp92: causal local-level/local-trend state-space Cost signals."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_92';EXPERIMENT_NAME='Dynamic state-space Cost forecast';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=92
ALPHA=.35;BETA=.08

def _state(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x=x.sort_values(['canonical_project_id','snapshot_date']).copy();level=np.full(len(x),np.nan);trend=np.full(len(x),np.nan);innov=np.full(len(x),np.nan);var=np.full(len(x),np.nan)
    for _,idx in x.groupby('canonical_project_id',sort=False).groups.items():
        l=t=0.0;v=0.0;n=0
        for i in idx:
            z=pd.to_numeric(pd.Series([x.at[i,'cost_escalation_percentage']]),errors='coerce').iloc[0]
            if pd.isna(z): continue
            if n==0: l=float(z);t=0.0;e=0.0
            else:
                pred=l+t;e=float(z)-pred;new_l=pred+ALPHA*e;t=t+BETA*e;l=new_l;v=.8*v+.2*e*e
            level[i]=l;trend[i]=t;innov[i]=e;var[i]=np.sqrt(max(v,0));n+=1
    x['exp92_level']=level;x['exp92_trend']=trend;x['exp92_innovation']=innov;x['exp92_innovation_sd']=var;x['exp92_terminal_projection']=pd.Series(level,index=x.index)+6*pd.Series(trend,index=x.index);return x.sort_index()
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=_state(current_cost_oof(ctx['train'],ctx['cost_model']));score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];score=_state(score);features=['production_prediction','exp92_level','exp92_trend','exp92_innovation','exp92_innovation_sd','exp92_terminal_projection','duration_ratio','expenditure_ratio','progress_deviation'];corr,meta=fit_residual_booster(oof,score,features,9201);meta.update({'state_alpha':ALPHA,'state_beta':BETA,'state_filter':'causal local linear trend'});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
