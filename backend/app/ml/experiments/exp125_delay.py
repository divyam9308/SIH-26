"""Exp125: causal recent-trajectory shape encoder."""
import argparse,pandas as pd
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp125';NAME='Project trajectory shape encoder';SIGNALS=['schedule_slippage_days','physical_progress','expenditure_ratio','progress_deviation','cost_escalation_percentage']
def engineer(f):
    x=f.sort_values(['canonical_project_id','snapshot_date']).copy();g=x.groupby('canonical_project_id',sort=False)
    for c in SIGNALS:
        x[f'exp125_{c}_d1']=g[c].diff();x[f'exp125_{c}_d3']=g[c].diff(3);x[f'exp125_{c}_vol6']=g[c].transform(lambda z:pd.to_numeric(z,errors='coerce').rolling(6,min_periods=2).std())
    return x.sort_index()
def fit_experiment(end,output):
    c=prepare_context(end);o=engineer(production_oof(c));s=c['cohort'].copy();s['production_prediction']=c['production_delay'];se=engineer(pd.concat([c['train'],s],axis=0)).loc[s.index].copy();se['production_prediction']=c['production_delay'];feats=['production_prediction']+[f'exp125_{c}_{k}' for c in SIGNALS for k in ['d1','d3','vol6']]+['duration_ratio'];corr,d=fit_residual(o,se,feats,12501);d['trajectory_signals']=SIGNALS;return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
