"""Exp129: landmark multi-horizon remaining-time model."""
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp129';NAME='Landmark multi-horizon remaining-time model'
BASE=['duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage','approved_cost_cr'];H=[180,365,730,1095]
def attach(train,score):
    _,_,xt,xs=numeric_design(train,score,BASE);rem=(pd.to_datetime(train['completion_date'])-pd.to_datetime(train['snapshot_date'])).dt.days.to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);out=score.copy();probs=[]
    for h in H:
        m=LGBMClassifier(n_estimators=120,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=60,reg_alpha=5,reg_lambda=25,random_state=12900+h,verbosity=-1,n_jobs=1);m.fit(xt,(rem<=h).astype(int),sample_weight=w);p=m.predict_proba(xs)[:,1];probs.append(p);out[f'exp129_p_{h}']=p
    P=np.vstack(probs);out['exp129_long_tail']=1-P[-1];out['exp129_entropy']=np.mean(-(P*np.log(np.clip(P,1e-6,1))+(1-P)*np.log(np.clip(1-P,1e-6,1))),axis=0);return out
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);yc=pd.to_numeric(o['oof_year'],errors='coerce');parts=[]
    for y in sorted(int(x) for x in yc.dropna().unique())[1:]:
        f=o.loc[yc<y].copy();v=o.loc[yc==y].copy()
        if len(f)>=100 and not v.empty:parts.append(attach(f,v))
    meta=pd.concat(parts,ignore_index=True);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];s=attach(o,s);features=['production_prediction']+[f'exp129_p_{h}' for h in H]+['exp129_long_tail','exp129_entropy']+BASE;corr,d=fit_residual(meta,s,features,12901);d['horizons_days']=H;return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
