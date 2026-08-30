"""Experiment 70 / U9: causal temporal-consistency filter for Exp61 Delay."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import delay_oof_frame,production_comparison,run_cli

EXPERIMENT_ID="exp_70";EXPERIMENT_SEQUENCE=70;MARKER="EXP70"
EXPERIMENT_NAME="U9 causal completion-date consistency filter";EXPERIMENT_SCOPE="delay"
CHANGED_DIMENSION="training_selected_causal_completion_date_smoothing"
ALPHAS=(0.25,0.5,0.75,1.0)


def causal_completion_filter(frame: pd.DataFrame, predictions, alpha: float):
    work=frame.copy();work["_raw"]=np.asarray(predictions,float);work["snapshot_date"]=pd.to_datetime(work["snapshot_date"],errors="coerce");planned=pd.to_datetime(work.get("planned_completion_date"),errors="coerce");work["_planned"]=planned;work["_implied"]=(planned+pd.to_timedelta(work["_raw"],unit="D")).astype("int64",errors="ignore") if False else np.nan
    valid=planned.notna()&np.isfinite(work["_raw"].to_numpy(float));work.loc[valid,"_implied"]=planned.loc[valid].astype("int64").to_numpy(float)/86400e9+work.loc[valid,"_raw"].to_numpy(float)
    result=work["_raw"].to_numpy(float).copy();order=work.sort_values(["canonical_project_id","snapshot_date"],kind="mergesort").index
    for _,idx in work.loc[order].groupby("canonical_project_id",sort=False).groups.items():
        prev=None
        for label in idx:
            pos=work.index.get_loc(label);cur=work.at[label,"_implied"]
            if not np.isfinite(cur): continue
            filt=cur if prev is None else alpha*cur+(1-alpha)*prev;prev=filt;p=work.at[label,"_planned"]
            result[pos]=max(0.0,filt-(pd.Timestamp(p).value/86400e9))
    return result


def _weighted_mae(actual,pred,w):
    a=np.asarray(actual,float);p=np.asarray(pred,float);w=np.asarray(w,float);m=np.isfinite(a)&np.isfinite(p)&np.isfinite(w)&(w>=0);return float(np.average(np.abs(a[m]-p[m]),weights=w[m]))


def select_alpha(oof: pd.DataFrame):
    raw=pd.to_numeric(oof["production_prediction"],errors="coerce").to_numpy(float);actual=pd.to_numeric(oof["actual_delay_days"],errors="coerce").to_numpy(float);w=pd.to_numeric(oof["sample_weight"],errors="coerce").fillna(0).to_numpy(float);scores={}
    for alpha in ALPHAS: scores[alpha]=_weighted_mae(actual,causal_completion_filter(oof,raw,alpha),w)
    best=min(ALPHAS,key=lambda a:(scores[a],-a));return float(best),{str(a):scores[a] for a in ALPHAS}


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,c,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end);oof=delay_oof_frame(data,production_bundle,training_start,training_end,test_end);alpha,scores=select_alpha(oof);ed=causal_completion_filter(c,pdly,alpha)
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,pc.copy(),pdly,ed,{"baseline":"assumed Exp61 production from PR #96","selected_alpha":alpha,"training_oof_alpha_mae":scores,"alpha_candidates":list(ALPHAS),"causal_current_and_prior_predictions_only":True,"cost_predictions_identical":True})

if __name__=="__main__":run_cli(sys.modules[__name__])
