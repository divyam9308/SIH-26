"""Exp126: monotonic constrained residual model."""
import argparse
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp126';NAME='Monotonic Delay residual model'
FEATS=['production_prediction','schedule_slippage_days','duration_ratio','physical_progress','expenditure_ratio','progress_deviation','cost_escalation_percentage'];MONO={'schedule_slippage_days':1,'duration_ratio':1,'physical_progress':-1,'expenditure_ratio':-1}
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];corr,d=fit_residual(o,s,FEATS,12601,monotone=MONO);d['monotone_constraints']=MONO;return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
