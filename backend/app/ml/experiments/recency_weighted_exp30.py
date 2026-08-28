"""Experiment 30: recency-weighted temporal training."""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED, _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract

EXPERIMENT_ID="exp_30"; EXPERIMENT_NAME="Recency-weighted temporal training"; EXPERIMENT_SCOPE="cost+delay"; EXPERIMENT_SEQUENCE=30
DELAY_SEED=26204; HALF_LIVES=[None,4.0,8.0,12.0]; MAX_FOLDS=3

def _gain(base,candidate): return (base-candidate)/base*100.0 if base else 0.0
def _key(row): return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()

def _apply_recency_weights(frame, reference_year, half_life):
    weighted=frame.copy()
    if half_life is None: return weighted
    years=pd.to_numeric(weighted.completion_year, errors="coerce"); age=(float(reference_year)-years).clip(lower=0)
    multiplier=np.power(0.5, age/float(half_life))
    weighted["sample_weight"]=pd.to_numeric(weighted.sample_weight, errors="coerce").fillna(0)*multiplier
    total=float(weighted.sample_weight.sum()); original=float(pd.to_numeric(frame.sample_weight, errors="coerce").fillna(0).sum())
    if total<=0: raise ValueError("Experiment 30 produced zero training weight.")
    weighted["sample_weight"]*=original/total
    return weighted

def _rolling_folds(train):
    years=sorted(int(y) for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique()); folds=[]
    for year in reversed(years[1:]):
        fitting=train[pd.to_numeric(train.completion_year, errors="coerce").lt(year)].copy(); validation=train[pd.to_numeric(train.completion_year, errors="coerce").eq(year)].copy()
        if fitting.canonical_project_id.nunique()>=10 and validation.canonical_project_id.nunique()>=3: folds.append((fitting,validation,year))
        if len(folds)>=MAX_FOLDS: break
    return list(reversed(folds))

def _select_half_life(train,features,target,algorithm,seed):
    folds=_rolling_folds(train)
    if len(folds)<2: raise ValueError("Experiment 30 requires at least two rolling-origin folds.")
    comparisons=[]
    for half_life in HALF_LIVES:
        fold_rows=[]
        for fitting,validation,year in folds:
            model=_fit_pipeline(_regressors(seed)[algorithm], _apply_recency_weights(fitting,year-1,half_life), features,target)
            pred=model.predict(validation[features]); pred=np.maximum(0,pred) if target=="actual_delay_days" else pred
            metrics=_regression_metrics(validation[target],pred,validation.sample_weight,validation.canonical_project_id)
            fold_rows.append({"year":year,"MAE":float(metrics["MAE"])})
        maes=[r["MAE"] for r in fold_rows]; comparisons.append({"half_life_years":half_life,"mean_MAE":float(np.mean(maes)),"worst_MAE":float(np.max(maes)),"folds":fold_rows})
    winner=min(comparisons,key=lambda r:(r["mean_MAE"],r["worst_MAE"]))
    for r in comparisons: r["selected"]=r is winner
    return winner["half_life_years"],comparisons

def fit_experiment(*,data,training_start,training_end,test_end,production_bundle,production_receipt,**_):
    enriched=enrich_supervised_for_production(data.copy()); enriched["completion_year"]=pd.to_numeric(enriched.completion_year,errors="coerce"); enriched["snapshot_date"]=pd.to_datetime(enriched.snapshot_date,errors="coerce")
    train,test=temporal_project_split(enriched,training_start,training_end,test_end)
    metadata=dict(production_bundle.get("metadata") or {}); contract=target_feature_contract(metadata); selected=dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    cost_name,delay_name=selected.get("cost"),selected.get("delay"); cost_features,delay_features=list(contract["cost"]),list(contract["delay"])
    cost_half,cost_sel=_select_half_life(train,cost_features,"actual_cost_overrun_percentage",cost_name,PRODUCTION_COST_SEED)
    delay_half,delay_sel=_select_half_life(train,delay_features,"actual_delay_days",delay_name,DELAY_SEED)
    cost_model=production_bundle["cost"] if cost_half is None else _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[cost_name],_apply_recency_weights(train,training_end,cost_half),cost_features,"actual_cost_overrun_percentage")
    delay_model=production_bundle["delay"] if delay_half is None else _fit_pipeline(_regressors(DELAY_SEED)[delay_name],_apply_recency_weights(train,training_end,delay_half),delay_features,"actual_delay_days")
    cost_compare=_production_cost_evaluation_rows(test); prod_cost_pred=production_bundle["cost"].predict(cost_compare[cost_features]); exp_cost_pred=cost_model.predict(cost_compare[cost_features])
    prod_cost=_regression_metrics(cost_compare.actual_cost_overrun_percentage,prod_cost_pred,cost_compare.sample_weight,cost_compare.canonical_project_id); exp_cost=_regression_metrics(cost_compare.actual_cost_overrun_percentage,exp_cost_pred,cost_compare.sample_weight,cost_compare.canonical_project_id)
    prod_delay_pred=np.maximum(0,production_bundle["delay"].predict(test[delay_features])); exp_delay_pred=np.maximum(0,delay_model.predict(test[delay_features]))
    prod_delay=_regression_metrics(test.actual_delay_days,prod_delay_pred,test.sample_weight,test.canonical_project_id); exp_delay=_regression_metrics(test.actual_delay_days,exp_delay_pred,test.sample_weight,test.canonical_project_id)
    cg=_gain(float(prod_cost["MAE"]),float(exp_cost["MAE"])); dg=_gain(float(prod_delay["MAE"]),float(exp_delay["MAE"])); verdict="PROMOTION CANDIDATE" if cg>=0 and dg>=0 and (cg>0 or dg>0) else "REGRESSION / DO NOT PROMOTE"
    union=list(dict.fromkeys(cost_features+delay_features)); lookup={_key(r):{n:r.get(n) for n in union} for _,r in test.iterrows()}
    return {"experiment":{"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":EXPERIMENT_SCOPE,"run_id":f"exp30-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}","model_role":"experiment","promotion_allowed":False,"changed_dimension":"training_sample_weight_recency","selected_algorithms":selected,"selected_half_lives":{"cost":cost_half,"delay":delay_half},"internal_half_life_comparisons":{"cost":cost_sel,"delay":delay_sel},"future_holdout_used_for_selection":False,"decision":verdict},"overall_comparison":{"production_cost_mae":prod_cost["MAE"],"experiment_cost_mae":exp_cost["MAE"],"cost_improvement_percentage":round(cg,4),"production_delay_mae":prod_delay["MAE"],"experiment_delay_mae":exp_delay["MAE"],"delay_improvement_percentage":round(dg,4),"comparison_test_projects":int(test.canonical_project_id.nunique()),"comparison_test_snapshots":int(len(test)),"cost_comparison_projects":int(cost_compare.canonical_project_id.nunique()),"selected_cost_half_life_years":cost_half,"selected_delay_half_life_years":delay_half,"decision":verdict},"runtime_state":{"cost_model":cost_model,"delay_model":delay_model,"cost_features":cost_features,"delay_features":delay_features,"lookup":lookup,"comparable":set(lookup)}}

def filter_comparable_rows(frame,state): return frame[frame.apply(lambda row:_key(row) in state["comparable"],axis=1)].copy()
def predict_project(row,state):
    key=_key(row)
    if key not in state["lookup"]: raise ValueError("No Experiment 30 feature vector is available for this snapshot.")
    candidate=row.copy()
    for n,v in state["lookup"][key].items(): candidate[n]=v
    cost=float(state["cost_model"].predict(candidate.to_frame().T.reindex(columns=state["cost_features"]))[0]); delay=max(0.0,float(state["delay_model"].predict(candidate.to_frame().T.reindex(columns=state["delay_features"]))[0]))
    return {"predicted_cost_overrun":round(cost,4),"predicted_delay_days":round(delay,4)}
