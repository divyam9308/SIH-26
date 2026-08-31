"""Experiment 84: conservative high-confidence Cost tail uplift gate."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from backend.app.ml.experiments.post_u1_cost_common import _wq,current_cost_oof,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_84';EXPERIMENT_NAME='Tail-risk conservative Cost uplift gate';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=84
FEATURES=['production_prediction','cost_escalation_percentage','expenditure_ratio','duration_ratio','schedule_slippage_days','approved_cost_cr','revised_cost_cr','elapsed_days','planned_duration_days']
GATE=.80;SCALE=.25

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];cols,med,xo,xs=numeric_design(oof,score,FEATURES);r=pd.to_numeric(oof['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(oof['sample_weight'],errors='coerce').fillna(0).to_numpy(float);thr=max(0.0,_wq(r,w,.75));label=(r>thr).astype(int)
    if label.min()==label.max():
        prob=np.zeros(len(score),float);magnitude=0.0
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=8401));model.fit(xo,label,logisticregression__sample_weight=w);prob=model.predict_proba(xs)[:,1];magnitude=max(0.0,_wq(r[label==1],w[label==1],.5))
    active=prob>=GATE;corr=np.where(active,SCALE*prob*magnitude,0.0);pred=ctx['production_cost']+corr;details={'features':cols,'medians':med,'training_positive_residual_threshold':thr,'gate_probability':GATE,'correction_scale':SCALE,'positive_residual_median':magnitude,'activation_snapshots':int(active.sum()),'activation_fraction':float(active.mean()),'correction_is_positive_only':True};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
