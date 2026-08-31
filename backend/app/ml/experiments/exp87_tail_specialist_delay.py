"""Experiment 87: tail-specialized conservative U1 Delay correction."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from backend.app.ml.experiments.post_u1_delay_common import _wq,current_delay_oof,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_87';EXPERIMENT_NAME='Tail-specialized U1 Delay correction';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=87
FEATURES=['production_prediction','u1_correction','aft_disagreement','schedule_slippage_days','duration_ratio','expenditure_ratio','cost_escalation_percentage','approved_cost_cr','exp58_delay_hier_prior','exp58_group_support'];GATE=.70;SCALE=.50

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base
    cols,med,xo,xs=numeric_design(oof,score,FEATURES);r=pd.to_numeric(oof['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(oof['sample_weight'],errors='coerce').fillna(0).to_numpy(float);thr=max(_wq(np.abs(r),w,.75),1e-9);label=(np.abs(r)>=thr).astype(int)
    if label.min()==label.max() or int(label.sum())<50:
        corr=np.zeros(len(score),float);prob=np.zeros(len(score),float);active=np.zeros(len(score),bool)
    else:
        gate=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=8701));gate.fit(xo,label,sample_weight=w);prob=gate.predict_proba(xs)[:,1];tail=oof.loc[label.astype(bool)].copy();_,_,xt,xscore=numeric_design(tail,score,FEATURES);m=LGBMRegressor(n_estimators=120,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=40,reg_alpha=5,reg_lambda=25,random_state=8702,verbosity=-1,n_jobs=1);rr=pd.to_numeric(tail['residual'],errors='coerce').fillna(0).to_numpy(float);ww=pd.to_numeric(tail['sample_weight'],errors='coerce').fillna(0).to_numpy(float);m.fit(xt,rr,sample_weight=ww);cap=max(_wq(np.abs(rr),ww,.9),1e-9);special=np.clip(np.asarray(m.predict(xscore),float),-cap,cap);active=prob>=GATE;corr=np.where(active,SCALE*prob*special,0.0)
    details={'features':cols,'medians':med,'tail_abs_residual_q75':thr,'gate_probability':GATE,'specialist_scale':SCALE,'activation_snapshots':int(active.sum()),'activation_fraction':float(active.mean()),'tail_training_rows':int(label.sum())};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
