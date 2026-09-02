"""Experiment I: temporal revised-budget credibility for Cost.

For each historical completed project, measure how far final completion expenditure
ended above/below the revised budget that was visible during execution.  Priors
are shrunk by agency/sector and training rows receive only strictly earlier-year
priors.  The future holdout receives priors fit on the allowed training archive.
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

EXPERIMENT_ID="exp_i"
EXPERIMENT_NAME="I — revised-budget credibility prior"
SEED=13801
STRENGTH=20.0
BASE_FEATURES=["approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","expenditure_ratio","physical_progress","duration_ratio","schedule_slippage_days","progress_deviation","cost_escalation_percentage","exp12_cost_velocity_12m","exp12_expenditure_velocity_6m","exp34_cost_revision_count","exp34_cumulative_abs_cost_revision_pct"]
PRIOR_FEATURES=["exp_i_budget_bias_pct","exp_i_budget_bias_support","exp_i_adjusted_current_overrun_pct","exp_i_revised_to_approved_ratio"]


def _num(f,c): return pd.to_numeric(f.get(c,pd.Series(np.nan,index=f.index)),errors="coerce")
def _norm(f,c):
    k="_norm_"+c
    if k in f:return f[k].astype("string").fillna("<NA>")
    return f.get(c,pd.Series("<NA>",index=f.index)).astype("string").fillna("<NA>").str.lower().str.replace(r"[^a-z0-9]+"," ",regex=True).str.strip().replace("","<NA>")


def _project_bias(reference:pd.DataFrame):
    r=reference.copy();approved=_num(r,"approved_cost_cr");revised=_num(r,"revised_cost_cr");final=_num(r,"reported_completion_expenditure_cr");r["_bias"]=(final-revised)/approved.where(approved>0)*100;r["_agency"]=_norm(r,"implementing_agency");r["_sector"]=_norm(r,"sector");v=r.dropna(subset=["canonical_project_id","_bias"])
    if v.empty:return pd.DataFrame(columns=["canonical_project_id","bias","agency","sector"])
    return v.groupby("canonical_project_id",as_index=False).agg(bias=("_bias","median"),agency=("_agency","last"),sector=("_sector","last"))


def _fit_prior(reference:pd.DataFrame):
    p=_project_bias(reference)
    if p.empty:return {"global":0.0,"maps":{},"projects":0}
    global_bias=float(p["bias"].median());maps={}
    for keys in [("agency","sector"),("sector",),("agency",)]:
        g=p.groupby(list(keys),dropna=False)["bias"].agg(["median","count"]).reset_index();g["value"]=(g["count"]*g["median"]+STRENGTH*global_bias)/(g["count"]+STRENGTH);maps[keys]=g
    return {"global":global_bias,"maps":maps,"projects":len(p)}


def attach_budget_credibility(reference:pd.DataFrame,score:pd.DataFrame):
    prior=_fit_prior(reference);out=score.copy();out["_row"]=np.arange(len(out));out["_agency"]=_norm(out,"implementing_agency");out["_sector"]=_norm(out,"sector");values=np.full(len(out),float(prior["global"]));support=np.zeros(len(out));unresolved=np.ones(len(out),bool)
    for keys in [("agency","sector"),("sector",),("agency",)]:
        lookup=prior["maps"].get(keys)
        if lookup is None:continue
        left=["_"+k for k in keys];right=lookup.rename(columns={k:"_"+k for k in keys});m=out[left].merge(right,on=left,how="left",sort=False);found=m["value"].notna().to_numpy()&unresolved;values[found]=m.loc[found,"value"].to_numpy(float);support[found]=m.loc[found,"count"].to_numpy(float);unresolved[found]=False
    out["exp_i_budget_bias_pct"]=values;out["exp_i_budget_bias_support"]=support;out["exp_i_adjusted_current_overrun_pct"]=_num(out,"cost_escalation_percentage").fillna(0)+values;out["exp_i_revised_to_approved_ratio"]=_num(out,"revised_cost_cr")/_num(out,"approved_cost_cr").replace(0,np.nan);return out.sort_values("_row",kind="mergesort").drop(columns=["_row","_agency","_sector"],errors="ignore")


def crossfit_training_priors(train:pd.DataFrame):
    out=train.copy();out["_row"]=np.arange(len(out));years=pd.to_numeric(out["completion_year"],errors="coerce");parts=[]
    for year in sorted(int(v) for v in years.dropna().unique()):
        val=out.loc[years==year].copy();ref=out.loc[years<year].copy()
        if ref["canonical_project_id"].nunique()<10:
            for c in PRIOR_FEATURES: val[c]=0.0
        else: val=attach_budget_credibility(ref,val)
        parts.append(val)
    if not parts:raise ValueError("Experiment I has no temporal training folds")
    return pd.concat(parts,ignore_index=True).sort_values("_row",kind="mergesort").drop(columns=["_row"],errors="ignore")


def _design(train,score,features):
    cols=[];A={};B={}
    for c in features:
        if c not in train or c not in score:continue
        a=_num(train,c).replace([np.inf,-np.inf],np.nan);b=_num(score,c).replace([np.inf,-np.inf],np.nan)
        if not a.notna().any():continue
        med=float(a.median());cols.append(c);A[c]=a.fillna(med);B[c]=b.fillna(med)
    return cols,pd.DataFrame(A,index=train.index),pd.DataFrame(B,index=score.index)

def _model(seed):return LGBMRegressor(objective="regression_l1",n_estimators=450,learning_rate=.025,max_depth=4,num_leaves=16,min_child_samples=60,subsample=.9,colsample_bytree=.9,reg_alpha=4,reg_lambda=20,random_state=seed,verbosity=-1,n_jobs=2)
def _fit_predict(train,score,features,seed):
    cols,xt,xs=_design(train,score,features);y=_num(train,"actual_cost_overrun_percentage");v=y.notna();m=_model(seed);m.fit(xt.loc[v],y.loc[v].to_numpy(float),sample_weight=_num(train.loc[v],"sample_weight").fillna(0).to_numpy(float));return np.asarray(m.predict(xs),float),cols


def fit_experiment(training_end:int,output:str):
    ctx=prepare_context(training_end);raw_train=ctx["train"].copy();train=crossfit_training_priors(raw_train);score=attach_budget_credibility(raw_train,ctx["cohort"].copy());control,control_cols=_fit_predict(train,score,BASE_FEATURES,SEED);candidate,candidate_cols=_fit_predict(train,score,BASE_FEATURES+PRIOR_FEATURES,SEED)
    pc=np.asarray(ctx["production_cost"],float);pdly=np.asarray(ctx["production_delay"],float);pcm=metric(score,"actual_cost_overrun_percentage",pc);ecm=metric(score,"actual_cost_overrun_percentage",candidate);ctrl=metric(score,"actual_cost_overrun_percentage",control);pdm=metric(score,"actual_delay_days",pdly);g=gain(pcm,ecm);ev=score[["canonical_project_id","sample_weight","actual_cost_overrun_percentage"]].copy();ev["production"]=pc;ev["experiment"]=candidate;boot=paired_project_mae_comparison(ev,actual="actual_cost_overrun_percentage",baseline_prediction="production",candidate_prediction="experiment",bootstrap_samples=5000,seed=SEED)
    result={"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":"cost","training_start":2001,"training_end":training_end,"test_start":ctx["test_start"],"test_end":ctx["test_end"],"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":round(g,6),"production_delay_mae":pdm,"experiment_delay_mae":pdm,"delay_improvement_percentage":0.0,"comparison_test_projects":int(score["canonical_project_id"].nunique()),"comparison_test_snapshots":len(score),"delay_predictions_identical":True,"holdout_used_for_selection":False,"promotion_allowed":False,"paired_project_cost_bootstrap":boot,"execution_verdict":"EXECUTION VALID","scientific_verdict":"PROMOTION CANDIDATE" if g>0 else "DO NOT PROMOTE","details":{"changed_dimension":"revised_budget_credibility","training_priors":"strictly earlier completion years","one_project_contribution_to_prior":True,"prior_strength":STRENGTH,"internal_same_model_control_mae":ctrl,"control_features":control_cols,"candidate_features":candidate_cols,"prior_features":PRIOR_FEATURES,"holdout_outcomes_used_for_prior":False}}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(result),indent=2,allow_nan=False)+"\n");print(f"EXP_I_PRODUCTION_COST_MAE={pcm:.6f}");print(f"EXP_I_CONTROL_COST_MAE={ctrl:.6f}");print(f"EXP_I_EXPERIMENT_COST_MAE={ecm:.6f}");return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--end",type=int,choices=[2021,2022],required=True);p.add_argument("--output",required=True);a=p.parse_args();fit_experiment(a.end,a.output)


if __name__=="__main__":main()
