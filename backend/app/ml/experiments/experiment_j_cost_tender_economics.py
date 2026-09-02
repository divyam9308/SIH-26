"""Experiment J: verified contract-award / tender economics for Cost.

The external evidence must be explicitly linked to canonical PAIMANA project IDs,
source-attributed, and dated.  A snapshot can see only awards with
award_date <= snapshot_date.  If no verified tender file is tracked, production
is retained exactly rather than inferring contract terms from project names.
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

EXPERIMENT_ID="exp_j"
EXPERIMENT_NAME="J — verified tender and contract-award economics"
SEED=13901
ROOT=Path(__file__).resolve().parents[4]
DEFAULT_TENDER_PATH=ROOT/"data"/"external"/"official_project_tender_economics.csv"
REQUIRED=["canonical_project_id","award_date","estimate_cr","award_value_cr","package_count","rebid_count","contractor_name","source_url"]
BASE_FEATURES=["approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","expenditure_ratio","physical_progress","duration_ratio","schedule_slippage_days","progress_deviation","cost_escalation_percentage","exp12_cost_velocity_12m","exp12_expenditure_velocity_6m","exp34_cost_revision_count"]
TENDER_FEATURES=["exp_j_award_discount_pct","exp_j_contract_age_days","exp_j_package_count","exp_j_rebid_count","exp_j_award_to_approved_ratio","exp_j_contractor_archive_support"]


def _num(f,c):return pd.to_numeric(f.get(c,pd.Series(np.nan,index=f.index)),errors="coerce")


def load_verified_tenders(path:Path=DEFAULT_TENDER_PATH):
    if not path.exists():return None
    x=pd.read_csv(path,dtype={"canonical_project_id":"string"});missing=[c for c in REQUIRED if c not in x]
    if missing:raise ValueError(f"Verified tender file missing columns: {missing}")
    if x["source_url"].isna().any() or not x["source_url"].astype(str).str.startswith("https://").all():raise ValueError("Every tender row must carry a source-attributed https URL")
    x["award_date"]=pd.to_datetime(x["award_date"],errors="coerce");x["estimate_cr"]=pd.to_numeric(x["estimate_cr"],errors="coerce");x["award_value_cr"]=pd.to_numeric(x["award_value_cr"],errors="coerce");x["package_count"]=pd.to_numeric(x["package_count"],errors="coerce");x["rebid_count"]=pd.to_numeric(x["rebid_count"],errors="coerce")
    if x["award_date"].isna().any() or (x["estimate_cr"]<=0).any() or x["award_value_cr"].isna().any():raise ValueError("Verified tender file contains invalid dated economics")
    x["_contractor_support"]=x.groupby(x["contractor_name"].astype("string"))["canonical_project_id"].transform("nunique").astype(float)
    return x.sort_values(["canonical_project_id","award_date"],kind="mergesort").copy()


def attach_tender_economics(score:pd.DataFrame,tenders:pd.DataFrame):
    out=score.copy();out["_row"]=np.arange(len(out));out["snapshot_date"]=pd.to_datetime(out["snapshot_date"],errors="coerce")
    discount=np.full(len(out),np.nan);age=np.full(len(out),np.nan);packages=np.full(len(out),np.nan);rebids=np.full(len(out),np.nan);award_ratio=np.full(len(out),np.nan);support=np.full(len(out),np.nan)
    groups={str(k):v for k,v in tenders.groupby(tenders["canonical_project_id"].astype("string"),sort=False)}
    approved=_num(out,"approved_cost_cr").to_numpy(float)
    for pos,(_,row) in enumerate(out.iterrows()):
        project=str(row.get("canonical_project_id"));snapshot=row.get("snapshot_date");g=groups.get(project)
        if g is None or pd.isna(snapshot):continue
        visible=g.loc[g["award_date"]<=snapshot]
        if visible.empty:continue
        latest=visible.iloc[-1];estimate=float(latest["estimate_cr"]);award=float(latest["award_value_cr"])
        discount[pos]=(award-estimate)/estimate*100.0
        age[pos]=float((snapshot-latest["award_date"]).days)
        packages[pos]=float(latest["package_count"]);rebids[pos]=float(latest["rebid_count"])
        award_ratio[pos]=award/approved[pos] if np.isfinite(approved[pos]) and approved[pos]>0 else np.nan
        support[pos]=float(latest["_contractor_support"])
    out["exp_j_award_discount_pct"]=discount;out["exp_j_contract_age_days"]=age;out["exp_j_package_count"]=packages;out["exp_j_rebid_count"]=rebids;out["exp_j_award_to_approved_ratio"]=award_ratio;out["exp_j_contractor_archive_support"]=support
    return out.sort_values("_row",kind="mergesort").drop(columns=["_row"],errors="ignore")


def _design(train,score,features):
    cols=[];A={};B={}
    for c in features:
        if c not in train or c not in score:continue
        a=_num(train,c).replace([np.inf,-np.inf],np.nan);b=_num(score,c).replace([np.inf,-np.inf],np.nan)
        if not a.notna().any():continue
        m=float(a.median());cols.append(c);A[c]=a.fillna(m);B[c]=b.fillna(m)
    return cols,pd.DataFrame(A,index=train.index),pd.DataFrame(B,index=score.index)


def _result(ctx,pred,details,output):
    c=ctx["cohort"];pc=np.asarray(ctx["production_cost"],float);pdly=np.asarray(ctx["production_delay"],float);ec=np.asarray(pred,float);pcm=metric(c,"actual_cost_overrun_percentage",pc);ecm=metric(c,"actual_cost_overrun_percentage",ec);pdm=metric(c,"actual_delay_days",pdly);g=gain(pcm,ecm);ev=c[["canonical_project_id","sample_weight","actual_cost_overrun_percentage"]].copy();ev["production"]=pc;ev["experiment"]=ec;boot=paired_project_mae_comparison(ev,actual="actual_cost_overrun_percentage",baseline_prediction="production",candidate_prediction="experiment",bootstrap_samples=5000,seed=SEED)
    x={"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":"cost","training_start":2001,"training_end":ctx["training_end"],"test_start":ctx["test_start"],"test_end":ctx["test_end"],"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":round(g,6),"production_delay_mae":pdm,"experiment_delay_mae":pdm,"delay_improvement_percentage":0.0,"comparison_test_projects":int(c["canonical_project_id"].nunique()),"comparison_test_snapshots":len(c),"delay_predictions_identical":True,"holdout_used_for_selection":False,"promotion_allowed":False,"paired_project_cost_bootstrap":boot,"execution_verdict":"EXECUTION VALID","scientific_verdict":"PROMOTION CANDIDATE" if g>0 else "DO NOT PROMOTE","details":_json_safe(details)};p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(x),indent=2,allow_nan=False)+"\n");return x


def fit_experiment(training_end:int,output:str,tender_path:Path=DEFAULT_TENDER_PATH):
    ctx=prepare_context(training_end);tenders=load_verified_tenders(tender_path)
    if tenders is None:return _result(ctx,ctx["production_cost"].copy(),{"changed_dimension":"verified_tender_economics","verified_external_data_available":False,"proxy_substitution_allowed":False,"required_path":str(tender_path.relative_to(ROOT) if tender_path.is_relative_to(ROOT) else tender_path),"reason":"No source-attributed canonical-project tender archive is tracked; production retained exactly."},output)
    train=attach_tender_economics(ctx["train"],tenders);score=attach_tender_economics(ctx["cohort"],tenders);cols,xt,xs=_design(train,score,BASE_FEATURES+TENDER_FEATURES);target=_num(train,"actual_cost_overrun_percentage");valid=target.notna();model=LGBMRegressor(objective="regression_l1",n_estimators=450,learning_rate=.025,max_depth=4,num_leaves=16,min_child_samples=60,subsample=.9,colsample_bytree=.9,reg_alpha=4,reg_lambda=20,random_state=SEED,verbosity=-1,n_jobs=2);model.fit(xt.loc[valid],target.loc[valid].to_numpy(float),sample_weight=_num(train.loc[valid],"sample_weight").fillna(0).to_numpy(float));pred=np.asarray(model.predict(xs),float)
    coverage=float(score[TENDER_FEATURES[0]].notna().mean()) if len(score) else 0.0
    return _result(ctx,pred,{"changed_dimension":"verified_tender_economics","verified_external_data_available":True,"tender_rows":len(tenders),"source_urls":int(tenders["source_url"].nunique()),"snapshot_coverage":coverage,"award_date_must_precede_snapshot":True,"features":cols,"tender_features":TENDER_FEATURES,"raw_project_name_matching_allowed":False},output)


def main():
    p=argparse.ArgumentParser();p.add_argument("--end",type=int,choices=[2021,2022],required=True);p.add_argument("--output",required=True);p.add_argument("--tender-path",type=Path,default=DEFAULT_TENDER_PATH);a=p.parse_args();fit_experiment(a.end,a.output,a.tender_path)


if __name__=="__main__":main()
