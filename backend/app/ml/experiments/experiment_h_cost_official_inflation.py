"""Experiment H: official construction-input inflation exposure for Cost.

The experiment accepts only a source-attributed official monthly price-index file.
No PAIMANA Cost-escalation proxy is substituted if that file is absent.  With no
verified external evidence, the challenger is exactly current production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.post_exp113_delay_common import gain, metric, prepare_context
from backend.app.ml.monthly_training import _json_safe

EXPERIMENT_ID="exp_h"
EXPERIMENT_NAME="H — official construction-input inflation exposure"
SEED=13701
ROOT=Path(__file__).resolve().parents[4]
DEFAULT_INDEX_PATH=ROOT/"data"/"external"/"official_construction_wpi_monthly.csv"
REQUIRED=["month","index_value","source_url"]
BASE_FEATURES=["approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","expenditure_ratio","physical_progress","duration_ratio","schedule_slippage_days","progress_deviation","cost_escalation_percentage","exp12_cost_velocity_12m","exp12_expenditure_velocity_6m","exp34_cost_revision_count"]
INFLATION_FEATURES=["exp_h_current_index","exp_h_inflation_12m_pct","exp_h_inflation_24m_pct","exp_h_inflation_acceleration","exp_h_cumulative_since_approval_pct","exp_h_approved_cost_inflation_exposure_cr","exp_h_real_cost_escalation_pct"]


def _num(f,c): return pd.to_numeric(f.get(c,pd.Series(np.nan,index=f.index)),errors="coerce")


def load_official_index(path:Path=DEFAULT_INDEX_PATH):
    if not path.exists(): return None
    x=pd.read_csv(path); missing=[c for c in REQUIRED if c not in x]
    if missing: raise ValueError(f"Official construction index missing columns: {missing}")
    if x["source_url"].isna().any() or not x["source_url"].astype(str).str.startswith("https://").all(): raise ValueError("Every index row must have a source-attributed https URL")
    x["month"]=pd.to_datetime(x["month"],errors="coerce").dt.to_period("M").dt.to_timestamp(); x["index_value"]=pd.to_numeric(x["index_value"],errors="coerce")
    if x["month"].isna().any() or x["index_value"].isna().any() or (x["index_value"]<=0).any(): raise ValueError("Invalid official construction index values")
    x=x.sort_values("month",kind="mergesort").drop_duplicates("month",keep="last").copy()
    x["_infl12"]=(x["index_value"]/x["index_value"].shift(12)-1)*100
    x["_infl24"]=(x["index_value"]/x["index_value"].shift(24)-1)*100
    x["_accel"]=x["_infl12"]-x["_infl12"].shift(6)
    return x


def attach_inflation(score:pd.DataFrame,index:pd.DataFrame):
    out=score.copy(); out["_month"]=pd.to_datetime(out["snapshot_date"],errors="coerce").dt.to_period("M").dt.to_timestamp(); out["_approval_month"]=pd.to_datetime(out.get("approval_date"),errors="coerce").dt.to_period("M").dt.to_timestamp()
    current=index.set_index("month")
    out["exp_h_current_index"]=out["_month"].map(current["index_value"])
    out["exp_h_inflation_12m_pct"]=out["_month"].map(current["_infl12"])
    out["exp_h_inflation_24m_pct"]=out["_month"].map(current["_infl24"])
    out["exp_h_inflation_acceleration"]=out["_month"].map(current["_accel"])
    approval_index=out["_approval_month"].map(current["index_value"])
    cumulative=(out["exp_h_current_index"]/approval_index-1)*100
    out["exp_h_cumulative_since_approval_pct"]=cumulative
    out["exp_h_approved_cost_inflation_exposure_cr"]=_num(out,"approved_cost_cr")*cumulative/100.0
    out["exp_h_real_cost_escalation_pct"]=_num(out,"cost_escalation_percentage")-cumulative
    return out.drop(columns=["_month","_approval_month"],errors="ignore")


def _design(train,score,features):
    cols=[];A={};B={}
    for c in features:
        if c not in train or c not in score: continue
        a=_num(train,c).replace([np.inf,-np.inf],np.nan); b=_num(score,c).replace([np.inf,-np.inf],np.nan)
        if not a.notna().any(): continue
        m=float(a.median()); cols.append(c); A[c]=a.fillna(m); B[c]=b.fillna(m)
    return cols,pd.DataFrame(A,index=train.index),pd.DataFrame(B,index=score.index)


def _result(ctx,pred,details,output):
    c=ctx["cohort"];pc=np.asarray(ctx["production_cost"],float);pdly=np.asarray(ctx["production_delay"],float);ec=np.asarray(pred,float);pcm=metric(c,"actual_cost_overrun_percentage",pc);ecm=metric(c,"actual_cost_overrun_percentage",ec);pdm=metric(c,"actual_delay_days",pdly);g=gain(pcm,ecm)
    ev=c[["canonical_project_id","sample_weight","actual_cost_overrun_percentage"]].copy();ev["production"]=pc;ev["experiment"]=ec
    boot=paired_project_mae_comparison(ev,actual="actual_cost_overrun_percentage",baseline_prediction="production",candidate_prediction="experiment",bootstrap_samples=5000,seed=SEED)
    x={"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":"cost","training_start":2001,"training_end":ctx["training_end"],"test_start":ctx["test_start"],"test_end":ctx["test_end"],"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":round(g,6),"production_delay_mae":pdm,"experiment_delay_mae":pdm,"delay_improvement_percentage":0.0,"comparison_test_projects":int(c["canonical_project_id"].nunique()),"comparison_test_snapshots":len(c),"delay_predictions_identical":True,"holdout_used_for_selection":False,"promotion_allowed":False,"paired_project_cost_bootstrap":boot,"execution_verdict":"EXECUTION VALID","scientific_verdict":"PROMOTION CANDIDATE" if g>0 else "DO NOT PROMOTE","details":_json_safe(details)}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(x),indent=2,allow_nan=False)+"\n");return x


def fit_experiment(training_end:int,output:str,index_path:Path=DEFAULT_INDEX_PATH):
    ctx=prepare_context(training_end); index=load_official_index(index_path)
    if index is None:
        return _result(ctx,ctx["production_cost"].copy(),{"changed_dimension":"official_construction_inflation","verified_external_data_available":False,"proxy_substitution_allowed":False,"required_path":str(index_path.relative_to(ROOT) if index_path.is_relative_to(ROOT) else index_path),"reason":"No verified source-attributed official construction price index is tracked; production retained exactly."},output)
    train=attach_inflation(ctx["train"],index);score=attach_inflation(ctx["cohort"],index);cols,xt,xs=_design(train,score,BASE_FEATURES+INFLATION_FEATURES);target=_num(train,"actual_cost_overrun_percentage");valid=target.notna()
    model=LGBMRegressor(objective="regression_l1",n_estimators=450,learning_rate=.025,max_depth=4,num_leaves=16,min_child_samples=60,subsample=.9,colsample_bytree=.9,reg_alpha=4,reg_lambda=20,random_state=SEED,verbosity=-1,n_jobs=2);model.fit(xt.loc[valid],target.loc[valid].to_numpy(float),sample_weight=_num(train.loc[valid],"sample_weight").fillna(0).to_numpy(float));pred=np.asarray(model.predict(xs),float)
    return _result(ctx,pred,{"changed_dimension":"official_construction_inflation","verified_external_data_available":True,"index_rows":len(index),"source_urls":int(index["source_url"].nunique()),"features":cols,"inflation_features":INFLATION_FEATURES,"holdout_outcomes_used_for_index":False},output)


def main():
    p=argparse.ArgumentParser();p.add_argument("--end",type=int,choices=[2021,2022],required=True);p.add_argument("--output",required=True);p.add_argument("--index-path",type=Path,default=DEFAULT_INDEX_PATH);a=p.parse_args();fit_experiment(a.end,a.output,a.index_path)


if __name__=="__main__": main()
