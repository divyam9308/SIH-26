"""Exp128: cross-fitted successful-signal superensemble."""
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp128';NAME='Cross-fitted successful-model signal superensemble'
BASE=['production_prediction','duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage','approved_cost_cr']
def attach(train,score):
    _,_,xt,xs=numeric_design(train,score,BASE);y=pd.to_numeric(train['actual_delay_days'],errors='coerce').to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);out=score.copy();P=[]
    for i,(depth,leaves) in enumerate([(2,6),(3,8),(4,12)]):
        m=LGBMRegressor(n_estimators=140,learning_rate=.025,max_depth=depth,num_leaves=leaves,min_child_samples=70,reg_alpha=5,reg_lambda=30,random_state=12800+i,verbosity=-1,n_jobs=1);m.fit(xt,y,sample_weight=w);p=np.maximum(0,np.asarray(m.predict(xs),float));P.append(p);out[f'exp128_aux_{i}']=p
    A=np.vstack(P);out['exp128_aux_median']=np.median(A,axis=0);out['exp128_aux_disagreement']=np.std(A,axis=0);return out
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);yc=pd.to_numeric(o['oof_year'],errors='coerce');parts=[]
    for y in sorted(int(x) for x in yc.dropna().unique())[1:]:
        f=o.loc[yc<y].copy();v=o.loc[yc==y].copy()
        if len(f)>=100 and not v.empty:parts.append(attach(f,v))
    meta=pd.concat(parts,ignore_index=True);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];s=attach(o,s);features=BASE+['exp128_aux_0','exp128_aux_1','exp128_aux_2','exp128_aux_median','exp128_aux_disagreement'];corr,d=fit_residual(meta,s,features,12801);d['stack']='three regularized auxiliary Delay learners + production anchor';return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
