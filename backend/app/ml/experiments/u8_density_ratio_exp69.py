"""Experiment 69 / U8: training-only density-ratio temporal adaptation of Exp61."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.app.ml.experiments.exp35_aft_residual_combo import _aft_remaining_prediction,_corrections,_delay_aft_calibration_oof,_delay_from_remaining,_fit_aft_family_models,_remaining_frame
from backend.app.ml.experiments.nextgen_common import _compare,_cost_oof,_family,_persist,_prepare,normalize_taxonomy,shrunk_calibration
from backend.app.ml.experiments.post61_common import run_cli
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors

EXPERIMENT_ID="exp_69";EXPERIMENT_SEQUENCE=69;MARKER="EXP69"
EXPERIMENT_NAME="U8 density-ratio temporal adaptation";EXPERIMENT_SCOPE="cost+delay"
CHANGED_DIMENSION="recent_training_covariate_density_ratio_weights"
DENSITY_FEATURES=["approved_cost_cr","duration_ratio","cost_escalation_percentage","schedule_slippage_days","expenditure_ratio","progress_deviation"]


def density_project_weights(train: pd.DataFrame, training_end: int):
    ordered=train.sort_values(["canonical_project_id","snapshot_date"],kind="mergesort");latest=ordered.groupby("canonical_project_id",as_index=False).tail(1).copy();cols=[c for c in DENSITY_FEATURES if c in latest]
    x=pd.DataFrame({c:pd.to_numeric(latest[c],errors="coerce") for c in cols});med=x.median();x=x.fillna(med);label=(pd.to_numeric(latest["completion_year"],errors="coerce")>=training_end-3).astype(int).to_numpy();prior=float(label.mean())
    if len(np.unique(label))<2 or not cols:
        weight=np.ones(len(latest),float)
    else:
        scaler=StandardScaler().fit(x);model=LogisticRegression(C=.5,max_iter=1000,random_state=6901).fit(scaler.transform(x),label);p=np.clip(model.predict_proba(scaler.transform(x))[:,1],.02,.98);odds=p/(1-p);prior_odds=prior/(1-prior);weight=np.clip(odds/prior_odds,.25,4.0)
    mapping=dict(zip(latest["canonical_project_id"].astype("string"),weight));return mapping,{"features":cols,"recent_year_start":training_end-3,"recent_project_share":prior,"min_weight":float(np.min(weight)),"max_weight":float(np.max(weight)),"mean_weight":float(np.mean(weight)),"clip":[.25,4.0],"future_holdout_used":False}


def _apply_weights(frame,mapping):
    out=frame.copy();ratio=out["canonical_project_id"].astype("string").map(mapping).fillna(1.0).to_numpy(float);out["sample_weight"]=pd.to_numeric(out["sample_weight"],errors="coerce").fillna(0).to_numpy(float)*ratio;return out


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    frame=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(frame,training_start,training_end,test_end);mapping,details=density_project_weights(train,training_end);weighted_train=_apply_weights(train,mapping);prior_train,prior_test,_=_build_temporal_delay_priors(train,test);weighted_prior=_apply_weights(prior_train,mapping);c=_compare(prior_test)
    cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c),float);pdly=np.maximum(0,np.asarray(dm.predict(c),float))
    cfeat=list(cm.features);fam=_family(cm);cal=shrunk_calibration(_cost_oof(weighted_train,cfeat,fam),40.0);model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],weighted_train,cfeat,"actual_cost_overrun_percentage");raw=np.asarray(model.predict(c[cfeat]),float);ec=raw+_corrections(c,raw,cal)
    dfeat=list(dm.model_features);td=_remaining_frame(weighted_prior);dcal,_=_delay_aft_calibration_oof(td,dfeat,dm.weights);models=_fit_aft_family_models(td,dfeat);ed=pdly.copy();elig=AFTResidualDelayModel._aft_eligible(c).to_numpy(bool)
    if elig.any():
        pos=np.flatnonzero(elig);sub=c.iloc[pos];rem=_aft_remaining_prediction(models,dm.weights,sub,dfeat);draw=_delay_from_remaining(sub,rem);ed[pos]=np.maximum(0,draw+_corrections(sub,draw,dcal))
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,ec,pdly,ed,{"baseline":"assumed Exp61 production from PR #96","density_ratio":details,"fallback_delay_retained":True})

if __name__=="__main__":run_cli(sys.modules[__name__])
