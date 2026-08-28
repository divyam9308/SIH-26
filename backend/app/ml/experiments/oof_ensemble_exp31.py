"""Experiment 31: rolling-OOF non-negative multi-model MAE ensemble."""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from backend.app.ml.monthly_training import _fit_pipeline,_regression_metrics,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED,_production_cost_evaluation_rows,enrich_supervised_for_production,target_feature_contract

EXPERIMENT_ID="exp_31"; EXPERIMENT_NAME="Rolling-OOF non-negative MAE ensemble"; EXPERIMENT_SCOPE="cost+delay"; EXPERIMENT_SEQUENCE=31
DELAY_SEED=26204; MAX_FOLDS=3; GRID_STEP=0.1; FAMILIES=("extra_trees","lightgbm","xgboost")

def _gain(b,c): return (b-c)/b*100.0 if b else 0.0
def _key(r): return str(r.canonical_project_id),pd.Timestamp(r.snapshot_date).isoformat()

def _rolling_folds(train):
    years=sorted(int(y) for y in pd.to_numeric(train.completion_year,errors="coerce").dropna().unique()); folds=[]
    for year in reversed(years[1:]):
        fitting=train[pd.to_numeric(train.completion_year,errors="coerce").lt(year)].copy(); validation=train[pd.to_numeric(train.completion_year,errors="coerce").eq(year)].copy()
        if fitting.canonical_project_id.nunique()>=10 and validation.canonical_project_id.nunique()>=3: folds.append((fitting,validation,year))
        if len(folds)>=MAX_FOLDS: break
    return list(reversed(folds))

def _weight_grid(step=GRID_STEP):
    units=int(round(1.0/step)); rows=[]
    for a in range(units+1):
        for b in range(units+1-a):
            c=units-a-b; rows.append({FAMILIES[0]:a/units,FAMILIES[1]:b/units,FAMILIES[2]:c/units})
    return rows

def _oof_weights(train,features,target,seed):
    folds=_rolling_folds(train)
    if len(folds)<2: raise ValueError("Experiment 31 requires at least two rolling OOF folds.")
    chunks=[]; diagnostics=[]
    for fitting,validation,year in folds:
        chunk=validation[[target,"sample_weight","canonical_project_id"]].copy(); diag={"year":year,"projects":int(validation.canonical_project_id.nunique())}
        for family in FAMILIES:
            model=_fit_pipeline(_regressors(seed)[family],fitting,features,target); pred=model.predict(validation[features]); pred=np.maximum(0,pred) if target=="actual_delay_days" else pred
            chunk[family]=pred; diag[family]=_regression_metrics(validation[target],pred,validation.sample_weight,validation.canonical_project_id)["MAE"]
        chunks.append(chunk); diagnostics.append(diag)
    oof=pd.concat(chunks,ignore_index=True); actual=pd.to_numeric(oof[target],errors="coerce").to_numpy(float); weights=pd.to_numeric(oof.sample_weight,errors="coerce").to_numpy(float)
    best=None; comparisons=[]
    for blend in _weight_grid():
        pred=sum(float(blend[name])*oof[name].to_numpy(float) for name in FAMILIES); mae=float(mean_absolute_error(actual,pred,sample_weight=weights)); row={"weights":blend,"MAE":mae}; comparisons.append(row)
        if best is None or mae<best["MAE"]: best=row
    return dict(best["weights"]),comparisons,{"folds":diagnostics,"oof_rows":int(len(oof))}

def _fit_family_models(train,features,target,seed): return {f:_fit_pipeline(_regressors(seed)[f],train,features,target) for f in FAMILIES}
def _blend_predict(models,weights,frame,features):
    pred=np.zeros(len(frame),dtype=float)
    for family in FAMILIES: pred+=float(weights[family])*models[family].predict(frame[features])
    return pred

def fit_experiment(*,data,training_start,training_end,test_end,production_bundle,production_receipt,**_):
    enriched=enrich_supervised_for_production(data.copy()); enriched["completion_year"]=pd.to_numeric(enriched.completion_year,errors="coerce"); enriched["snapshot_date"]=pd.to_datetime(enriched.snapshot_date,errors="coerce")
    train,test=temporal_project_split(enriched,training_start,training_end,test_end); contract=target_feature_contract(dict(production_bundle.get("metadata") or {})); cost_features,delay_features=list(contract["cost"]),list(contract["delay"])
    cost_weights,cost_grid,cost_oof=_oof_weights(train,cost_features,"actual_cost_overrun_percentage",PRODUCTION_COST_SEED); delay_weights,delay_grid,delay_oof=_oof_weights(train,delay_features,"actual_delay_days",DELAY_SEED)
    cost_models=_fit_family_models(train,cost_features,"actual_cost_overrun_percentage",PRODUCTION_COST_SEED); delay_models=_fit_family_models(train,delay_features,"actual_delay_days",DELAY_SEED)
    cost_compare=_production_cost_evaluation_rows(test); prod_cost_pred=production_bundle["cost"].predict(cost_compare[cost_features]); exp_cost_pred=_blend_predict(cost_models,cost_weights,cost_compare,cost_features)
    prod_cost=_regression_metrics(cost_compare.actual_cost_overrun_percentage,prod_cost_pred,cost_compare.sample_weight,cost_compare.canonical_project_id); exp_cost=_regression_metrics(cost_compare.actual_cost_overrun_percentage,exp_cost_pred,cost_compare.sample_weight,cost_compare.canonical_project_id)
    prod_delay_pred=np.maximum(0,production_bundle["delay"].predict(test[delay_features])); exp_delay_pred=np.maximum(0,_blend_predict(delay_models,delay_weights,test,delay_features)); prod_delay=_regression_metrics(test.actual_delay_days,prod_delay_pred,test.sample_weight,test.canonical_project_id); exp_delay=_regression_metrics(test.actual_delay_days,exp_delay_pred,test.sample_weight,test.canonical_project_id)
    cg=_gain(float(prod_cost["MAE"]),float(exp_cost["MAE"])); dg=_gain(float(prod_delay["MAE"]),float(exp_delay["MAE"])); verdict="PROMOTION CANDIDATE" if cg>=0 and dg>=0 and (cg>0 or dg>0) else "REGRESSION / DO NOT PROMOTE"
    union=list(dict.fromkeys(cost_features+delay_features)); lookup={_key(r):{n:r.get(n) for n in union} for _,r in test.iterrows()}
    return {"experiment":{"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":EXPERIMENT_SCOPE,"run_id":f"exp31-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}","model_role":"experiment","promotion_allowed":False,"changed_dimension":"prediction_ensemble","families":list(FAMILIES),"selected_weights":{"cost":cost_weights,"delay":delay_weights},"rolling_oof":{"cost":cost_oof,"delay":delay_oof},"grid_size":{"cost":len(cost_grid),"delay":len(delay_grid)},"future_holdout_used_for_weight_selection":False,"decision":verdict},"overall_comparison":{"production_cost_mae":prod_cost["MAE"],"experiment_cost_mae":exp_cost["MAE"],"cost_improvement_percentage":round(cg,4),"production_delay_mae":prod_delay["MAE"],"experiment_delay_mae":exp_delay["MAE"],"delay_improvement_percentage":round(dg,4),"comparison_test_projects":int(test.canonical_project_id.nunique()),"comparison_test_snapshots":int(len(test)),"cost_comparison_projects":int(cost_compare.canonical_project_id.nunique()),"cost_blend_weights":cost_weights,"delay_blend_weights":delay_weights,"decision":verdict},"runtime_state":{"cost_models":cost_models,"delay_models":delay_models,"cost_weights":cost_weights,"delay_weights":delay_weights,"cost_features":cost_features,"delay_features":delay_features,"lookup":lookup,"comparable":set(lookup)}}

def filter_comparable_rows(frame,state): return frame[frame.apply(lambda row:_key(row) in state["comparable"],axis=1)].copy()
def predict_project(row,state):
    key=_key(row)
    if key not in state["lookup"]: raise ValueError("No Experiment 31 feature vector is available for this snapshot.")
    candidate=row.copy()
    for n,v in state["lookup"][key].items(): candidate[n]=v
    one=candidate.to_frame().T; cost=float(_blend_predict(state["cost_models"],state["cost_weights"],one,state["cost_features"])[0]); delay=max(0.0,float(_blend_predict(state["delay_models"],state["delay_weights"],one,state["delay_features"])[0])); return {"predicted_cost_overrun":round(cost,4),"predicted_delay_days":round(delay,4)}
