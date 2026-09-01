"""Exp122: discrete completion-hazard features."""
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp122';NAME='Discrete-time survival hazard stack'
BASE=['duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage','approved_cost_cr'];H=[180,365,730]
def attach(train,score):
    _,_,xt,xs=numeric_design(train,score,BASE);rem=(pd.to_datetime(train['completion_date'])-pd.to_datetime(train['snapshot_date'])).dt.days.to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);out=score.copy()
    for h in H:
        y=(rem<=h).astype(int);m=LGBMClassifier(n_estimators=120,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=60,reg_alpha=5,reg_lambda=25,random_state=12200+h,verbosity=-1,n_jobs=1);m.fit(xt,y,sample_weight=w);out[f'exp122_p_{h}']=m.predict_proba(xs)[:,1]
    out['exp122_hazard_entropy']=sum(-(out[f'exp122_p_{h}']*np.log(np.clip(out[f'exp122_p_{h}'],1e-6,1))+(1-out[f'exp122_p_{h}'])*np.log(np.clip(1-out[f'exp122_p_{h}'],1e-6,1))) for h in H);return out
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);yc=pd.to_numeric(o['oof_year'],errors='coerce');parts=[]
    for y in sorted(int(x) for x in yc.dropna().unique())[1:]:
        f=o.loc[yc<y].copy();v=o.loc[yc==y].copy()
        if len(f)>=100 and not v.empty:parts.append(attach(f,v))
    meta=pd.concat(parts,ignore_index=True);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];s=attach(o,s);feats=['production_prediction']+[f'exp122_p_{h}' for h in H]+['exp122_hazard_entropy']+BASE;corr,d=fit_residual(meta,s,feats,12201);d['horizons_days']=H;return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
