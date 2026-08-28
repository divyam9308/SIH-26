"""Experiment 32: AFT-style remaining-time forecasting (Delay only).

All supervised canonical training outcomes are completed projects, so the AFT
idea is tested in uncensored form: model log(1 + remaining days to completion)
from each as-of snapshot, transform back to remaining time, then derive final
delay relative to the planned completion date. Production Cost is retained.
"""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline,_regression_metrics,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows,enrich_supervised_for_production,target_feature_contract

EXPERIMENT_ID="exp_32"; EXPERIMENT_NAME="AFT-style remaining-time forecasting"; EXPERIMENT_SCOPE="delay"; EXPERIMENT_SEQUENCE=32; DELAY_SEED=26204

def _gain(b,c): return (b-c)/b*100.0 if b else 0.0
def _key(r): return str(r.canonical_project_id),pd.Timestamp(r.snapshot_date).isoformat()

def _remaining_frame(frame):
    out=frame.copy(); out["snapshot_date"]=pd.to_datetime(out.snapshot_date,errors="coerce"); out["completion_date"]=pd.to_datetime(out.get("completion_date"),errors="coerce"); out["planned_completion_date"]=pd.to_datetime(out.get("planned_completion_date"),errors="coerce")
    remaining=(out.completion_date-out.snapshot_date).dt.days; mask=remaining.gt(0)&out.planned_completion_date.notna(); out=out[mask].copy(); out["exp32_remaining_days"]=remaining[mask].astype(float); out["exp32_log_remaining_days"]=np.log1p(out.exp32_remaining_days); return assign_project_balanced_weights(out)

def _delay_from_remaining(frame,remaining_days):
    remaining=np.maximum(0.0,np.asarray(remaining_days,dtype=float)); snapshot=pd.to_datetime(frame.snapshot_date,errors="coerce"); planned=pd.to_datetime(frame.planned_completion_date,errors="coerce"); predicted=snapshot+pd.to_timedelta(remaining,unit="D"); return np.maximum(0.0,(predicted-planned).dt.total_seconds().to_numpy()/86400.0)

def fit_experiment(*,data,training_start,training_end,test_end,production_bundle,production_receipt,**_):
    enriched=enrich_supervised_for_production(data.copy()); enriched["completion_year"]=pd.to_numeric(enriched.completion_year,errors="coerce"); train,test=temporal_project_split(enriched,training_start,training_end,test_end); train_delay=_remaining_frame(train); compare=_remaining_frame(test)
    metadata=dict(production_bundle.get("metadata") or {}); contract=target_feature_contract(metadata); selected=dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {}); delay_name=selected.get("delay")
    if delay_name not in _regressors(DELAY_SEED): raise ValueError(f"Unsupported production Delay family for Exp32: {delay_name!r}")
    delay_features=list(contract["delay"]); model=_fit_pipeline(_regressors(DELAY_SEED)[delay_name],train_delay,delay_features,"exp32_log_remaining_days")
    prod_delay_pred=np.maximum(0,production_bundle["delay"].predict(compare[delay_features])); log_remaining=model.predict(compare[delay_features]); remaining_pred=np.expm1(np.clip(log_remaining,-20,20)); exp_delay_pred=_delay_from_remaining(compare,remaining_pred)
    prod_delay=_regression_metrics(compare.actual_delay_days,prod_delay_pred,compare.sample_weight,compare.canonical_project_id); exp_delay=_regression_metrics(compare.actual_delay_days,exp_delay_pred,compare.sample_weight,compare.canonical_project_id)
    cost_compare=_production_cost_evaluation_rows(test); cost_features=list(contract["cost"]); prod_cost_pred=production_bundle["cost"].predict(cost_compare[cost_features]); prod_cost=_regression_metrics(cost_compare.actual_cost_overrun_percentage,prod_cost_pred,cost_compare.sample_weight,cost_compare.canonical_project_id)
    dg=_gain(float(prod_delay["MAE"]),float(exp_delay["MAE"])); verdict="PROMOTION CANDIDATE" if dg>0 else "REGRESSION / DO NOT PROMOTE"; lookup={_key(r):{"snapshot_date":r.snapshot_date,"planned_completion_date":r.planned_completion_date} for _,r in compare.iterrows()}
    return {"experiment":{"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":EXPERIMENT_SCOPE,"run_id":f"exp32-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}","model_role":"experiment","promotion_allowed":False,"changed_dimension":"delay_target_formulation","target":"log1p(completion_date - snapshot_date)","censoring":"none: current supervised canonical outcomes are completed projects","selected_algorithms":selected,"cost_policy":"production_retained","future_holdout_used_for_training":False,"decision":verdict},"overall_comparison":{"production_cost_mae":prod_cost["MAE"],"experiment_cost_mae":prod_cost["MAE"],"cost_improvement_percentage":0.0,"production_delay_mae":prod_delay["MAE"],"experiment_delay_mae":exp_delay["MAE"],"delay_improvement_percentage":round(dg,4),"comparison_test_projects":int(compare.canonical_project_id.nunique()),"comparison_test_snapshots":int(len(compare)),"training_delay_projects":int(train_delay.canonical_project_id.nunique()),"training_delay_snapshots":int(len(train_delay)),"delay_comparison_filter":"positive remaining time and planned completion date available","decision":verdict},"runtime_state":{"delay_model":model,"delay_features":delay_features,"cost_model":production_bundle["cost"],"cost_features":cost_features,"lookup":lookup,"comparable":set(lookup)}}

def filter_comparable_rows(frame,state): return frame[frame.apply(lambda row:_key(row) in state["comparable"],axis=1)].copy()
def predict_project(row,state):
    key=_key(row)
    if key not in state["lookup"]: raise ValueError("Experiment 32 requires a snapshot with planned completion evidence.")
    candidate=row.copy(); candidate["snapshot_date"]=state["lookup"][key]["snapshot_date"]; candidate["planned_completion_date"]=state["lookup"][key]["planned_completion_date"]
    x=candidate.to_frame().T.reindex(columns=state["delay_features"]); remaining=float(np.expm1(np.clip(state["delay_model"].predict(x)[0],-20,20))); temp=pd.DataFrame([{"snapshot_date":candidate["snapshot_date"],"planned_completion_date":candidate["planned_completion_date"]}]); delay=float(_delay_from_remaining(temp,np.array([remaining]))[0]); cost=float(state["cost_model"].predict(candidate.to_frame().T.reindex(columns=state["cost_features"]))[0]); return {"predicted_cost_overrun":round(cost,4),"predicted_delay_days":round(delay,4),"predicted_remaining_days":round(max(0.0,remaining),4)}
