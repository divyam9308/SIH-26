"""Exp120: lifecycle-conditioned remaining-time residual experts."""
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp120';NAME='Lifecycle-conditioned remaining-time mixture of experts'
BASE=['production_prediction','duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage']
def stage(f):
    p=pd.to_numeric(f.get('physical_progress'),errors='coerce').fillna(0);d=pd.to_numeric(f.get('duration_ratio'),errors='coerce').fillna(0)
    return np.select([d>1.15,p<25,p<60,p<85],['overdue','early','mid','late'],default='finishing')
def enrich(oof,score):
    a=oof.copy();b=score.copy();a['exp120_stage']=stage(a);b['exp120_stage']=stage(b)
    for s in ['early','mid','late','finishing','overdue']:
        a[f'exp120_{s}']=(a['exp120_stage']==s).astype(float);b[f'exp120_{s}']=(b['exp120_stage']==s).astype(float)
    return a,b
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];o,s=enrich(o,s);feats=BASE+[f'exp120_{x}' for x in ['early','mid','late','finishing','overdue']];corr,d=fit_residual(o,s,feats,12001);d['stages']='early/mid/late/finishing/overdue';return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
