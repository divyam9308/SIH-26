"""Exp121: tail-error gated residual specialist."""
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp121';NAME='Quantile-AFT tail residual specialist'
FEATS=['production_prediction','duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage','approved_cost_cr']
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];thr=float(np.nanquantile(np.abs(pd.to_numeric(o['residual'],errors='coerce')),0.8));corr,d=fit_residual(o,s,FEATS,12100,extra_weight=lambda f:1+2*(np.abs(pd.to_numeric(f['residual'],errors='coerce').fillna(0).to_numpy(float))>=thr));d['tail_threshold_abs_oof_residual']=thr;return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
