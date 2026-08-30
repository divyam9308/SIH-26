"""Experiment 64 / U3: OOF production-error risk gate with residual specialist."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier

from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import cost_oof_frame,delay_oof_frame,production_comparison,run_cli,weighted_quantile

EXPERIMENT_ID="exp_64";EXPERIMENT_SEQUENCE=64;MARKER="EXP64"
EXPERIMENT_NAME="U3 production-error risk gate on Exp61";EXPERIMENT_SCOPE="cost+delay"
CHANGED_DIMENSION="large_oof_error_classifier_plus_residual_specialist"
FEATURES=["production_prediction","cost_escalation_percentage","schedule_slippage_days","duration_ratio","expenditure_ratio","progress_deviation","approved_cost_cr","exp58_delay_hier_prior","exp58_group_support"]


def _design(train,score):
    cols=[c for c in FEATURES if c in train and c in score]
    med={c:float(pd.to_numeric(train[c],errors="coerce").median()) for c in cols}
    a=pd.DataFrame({c:pd.to_numeric(train[c],errors="coerce").fillna(med[c]) for c in cols})
    b=pd.DataFrame({c:pd.to_numeric(score[c],errors="coerce").fillna(med[c]) for c in cols})
    return cols,a,b


def _fit_gate(oof,score,seed):
    cols,x,xs=_design(oof,score);w=pd.to_numeric(oof["sample_weight"],errors="coerce").fillna(0).to_numpy(float)
    r=pd.to_numeric(oof["residual"],errors="coerce").fillna(0).to_numpy(float);thr=weighted_quantile(np.abs(r),w,.75);label=(np.abs(r)>=thr).astype(int)
    if len(np.unique(label))<2:
        prob=np.full(len(xs),float(label[0]));gate=prob>=.5
    else:
        clf=RandomForestClassifier(n_estimators=220,max_depth=5,min_samples_leaf=25,class_weight="balanced_subsample",random_state=seed,n_jobs=2)
        clf.fit(x,label,sample_weight=w);prob=clf.predict_proba(xs)[:,1];gate=prob>=.5
    specialist_mask=label==1
    if specialist_mask.sum()<40: specialist_mask=np.ones(len(label),dtype=bool)
    reg=LGBMRegressor(n_estimators=160,learning_rate=.025,max_depth=3,num_leaves=12,min_child_samples=50,reg_alpha=5,reg_lambda=25,random_state=seed+100,verbosity=-1)
    reg.fit(x.loc[specialist_mask],r[specialist_mask],sample_weight=w[specialist_mask])
    corr=np.asarray(reg.predict(xs),float);cap=weighted_quantile(np.abs(r[specialist_mask]),w[specialist_mask],.90);corr=np.clip(corr,-cap,cap);corr=np.where(gate,corr,0.0)
    return corr,{"features":cols,"large_error_abs_threshold_q75":thr,"gate_threshold":.5,"routed_rows":int(gate.sum()),"oof_large_error_rows":int(label.sum()),"correction_cap_q90":cap,"holdout_tuned":False}


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,c,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end)
    co=cost_oof_frame(data,production_bundle,training_start,training_end,test_end);cs=c.copy();cs["production_prediction"]=pc;cc,cd=_fit_gate(co,cs,6401);ec=pc+cc
    do=delay_oof_frame(data,production_bundle,training_start,training_end,test_end);ds=c.copy();ds["production_prediction"]=pdly;dc,dd=_fit_gate(do,ds,6402);ed=np.maximum(0,pdly+dc)
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,ec,pdly,ed,{"baseline":"assumed Exp61 production from PR #96","cost":cd,"delay":dd,"base_prediction_retained_when_gate_false":True})

if __name__=="__main__":run_cli(sys.modules[__name__])
