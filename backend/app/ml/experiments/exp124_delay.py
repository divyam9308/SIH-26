"""Exp124: conformal uncertainty-gated correction."""
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp124';NAME='Conformal uncertainty-gated Delay correction'
FEATS=['production_prediction','duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage']
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];absr=np.abs(pd.to_numeric(o['residual'],errors='coerce').fillna(0));q50=float(absr.quantile(.5));q90=float(absr.quantile(.9));corr,d=fit_residual(o,s,FEATS,12401,extra_weight=lambda f:1+np.clip((np.abs(pd.to_numeric(f['residual'],errors='coerce').fillna(0).to_numpy(float))-q50)/max(q90-q50,1e-9),0,1));d.update({'conformal_abs_residual_q50':q50,'conformal_abs_residual_q90':q90});return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
