"""Experiment 68 / U7: event-centric past-only trajectory features on Exp61."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp35_aft_residual_combo import _aft_remaining_prediction,_corrections,_delay_aft_calibration_oof,_delay_from_remaining,_fit_aft_family_models,_remaining_frame
from backend.app.ml.experiments.nextgen_common import _compare,_cost_oof,_family,_persist,_prepare,normalize_taxonomy,shrunk_calibration
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.experiments.post61_common import run_cli

EXPERIMENT_ID="exp_68";EXPERIMENT_SEQUENCE=68;MARKER="EXP68"
EXPERIMENT_NAME="U7 event-centric trajectory features";EXPERIMENT_SCOPE="cost+delay"
CHANGED_DIMENSION="past_only_revision_shock_persistence_recovery_features"
EVENT_FEATURES=["exp68_cost_shock_count","exp68_schedule_shock_count","exp68_max_cost_jump","exp68_max_schedule_jump","exp68_months_since_cost_shock","exp68_months_since_schedule_shock","exp68_last_cost_jump","exp68_last_schedule_jump","exp68_cost_recovery_from_peak","exp68_schedule_recovery_from_peak"]


def enrich_events(frame: pd.DataFrame) -> pd.DataFrame:
    out=frame.copy();out["snapshot_date"]=pd.to_datetime(out["snapshot_date"],errors="coerce");out=out.sort_values(["canonical_project_id","snapshot_date"],kind="mergesort").copy();g=out.groupby("canonical_project_id",sort=False)
    cost=pd.to_numeric(out.get("cost_escalation_percentage"),errors="coerce");sched=pd.to_numeric(out.get("schedule_slippage_days"),errors="coerce")
    cdelta=g["cost_escalation_percentage"].diff() if "cost_escalation_percentage" in out else pd.Series(np.nan,index=out.index);sdelta=g["schedule_slippage_days"].diff() if "schedule_slippage_days" in out else pd.Series(np.nan,index=out.index)
    cshock=cdelta.abs().ge(5.0);sshock=sdelta.abs().ge(30.0)
    out["exp68_cost_shock_count"]=cshock.groupby(out["canonical_project_id"]).cumsum().astype(float);out["exp68_schedule_shock_count"]=sshock.groupby(out["canonical_project_id"]).cumsum().astype(float)
    out["exp68_max_cost_jump"]=cdelta.abs().groupby(out["canonical_project_id"]).cummax().fillna(0.0);out["exp68_max_schedule_jump"]=sdelta.abs().groupby(out["canonical_project_id"]).cummax().fillna(0.0)
    out["exp68_last_cost_jump"]=cdelta.fillna(0.0);out["exp68_last_schedule_jump"]=sdelta.fillna(0.0)
    cdate=out["snapshot_date"].where(cshock).groupby(out["canonical_project_id"]).ffill();sdate=out["snapshot_date"].where(sshock).groupby(out["canonical_project_id"]).ffill()
    out["exp68_months_since_cost_shock"]=(out["snapshot_date"]-cdate).dt.days/30.4375;out["exp68_months_since_schedule_shock"]=(out["snapshot_date"]-sdate).dt.days/30.4375
    cpeak=cost.groupby(out["canonical_project_id"]).cummax();speak=sched.groupby(out["canonical_project_id"]).cummax();out["exp68_cost_recovery_from_peak"]=(cpeak-cost).fillna(0.0);out["exp68_schedule_recovery_from_peak"]=(speak-sched).fillna(0.0)
    return out.sort_index()


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    frame=enrich_events(normalize_taxonomy(_prepare(data)));train,test=temporal_project_split(frame,training_start,training_end,test_end);prior_train,prior_test,_=_build_temporal_delay_priors(train,test);c=_compare(prior_test)
    cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c),float);pdly=np.maximum(0,np.asarray(dm.predict(c),float))
    cfeat=list(dict.fromkeys(list(cm.features)+EVENT_FEATURES));fam=_family(cm);cal=shrunk_calibration(_cost_oof(train,cfeat,fam),40.0);model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],train,cfeat,"actual_cost_overrun_percentage");raw=np.asarray(model.predict(c[cfeat]),float);ec=raw+_corrections(c,raw,cal)
    dfeat=list(dict.fromkeys(list(dm.model_features)+EVENT_FEATURES));td=_remaining_frame(prior_train);dcal,_=_delay_aft_calibration_oof(td,dfeat,dm.weights);models=_fit_aft_family_models(td,dfeat);ed=pdly.copy();elig=AFTResidualDelayModel._aft_eligible(c).to_numpy(bool)
    if elig.any():
        pos=np.flatnonzero(elig);sub=c.iloc[pos];remaining=_aft_remaining_prediction(models,dm.weights,sub,dfeat);draw=_delay_from_remaining(sub,remaining);ed[pos]=np.maximum(0,draw+_corrections(sub,draw,dcal))
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,ec,pdly,ed,{"baseline":"assumed Exp61 production from PR #96","event_features":EVENT_FEATURES,"cost_shock_threshold_pp":5.0,"schedule_shock_threshold_days":30.0,"aft_fallback_retained":True,"holdout_feature_selection":False})

if __name__=="__main__":run_cli(sys.modules[__name__])
