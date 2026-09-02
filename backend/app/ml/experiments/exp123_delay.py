"""Exp123: distribution stacking via heterogeneous remaining-time estimators."""
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp123';NAME='Remaining-time distribution stacking'
BASE=['duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage','approved_cost_cr']
def attach(train,score):
    _,_,xt,xs=numeric_design(train,score,BASE);rem=np.log1p(np.maximum(1,(pd.to_datetime(train['completion_date'])-pd.to_datetime(train['snapshot_date'])).dt.days.to_numpy(float)));w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);out=score.copy();pred=[]
    for i,loss in enumerate(['regression_l1','huber','quantile']):
        kw={'objective':loss};kw.update({'alpha':.5} if loss=='quantile' else {});m=LGBMRegressor(n_estimators=140,learning_rate=.025,max_depth=3,num_leaves=10,min_child_samples=70,reg_alpha=5,reg_lambda=25,random_state=12300+i,verbosity=-1,n_jobs=1,**kw);m.fit(xt,rem,sample_weight=w);r=np.maximum(1,np.expm1(np.asarray(m.predict(xs),float)));pred.append(r);out[f'exp123_remaining_{i}']=r
    A=np.vstack(pred);out['exp123_remaining_median']=np.median(A,axis=0);out['exp123_disagreement']=np.std(A,axis=0);return out
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);yc=pd.to_numeric(o['oof_year'],errors='coerce');parts=[]
    for y in sorted(int(x) for x in yc.dropna().unique())[1:]:
        f=o.loc[yc<y].copy();v=o.loc[yc==y].copy()
        if len(f)>=100 and not v.empty:parts.append(attach(f,v))
    meta=pd.concat(parts,ignore_index=True);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];s=attach(o,s);feats=['production_prediction','exp123_remaining_0','exp123_remaining_1','exp123_remaining_2','exp123_remaining_median','exp123_disagreement']+BASE;corr,d=fit_residual(meta,s,feats,12301);return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
