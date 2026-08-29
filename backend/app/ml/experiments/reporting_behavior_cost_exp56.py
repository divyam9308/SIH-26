"""Experiment 56 / C9: reporting behavior and missingness."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from backend.app.ml.experiments.nextgen_common import fit_features,run_cli
EXPERIMENT_ID="exp_56";EXPERIMENT_SEQUENCE=56;MARKER="EXP56";EXPERIMENT_NAME="Reporting behavior and missingness Cost features";EXPERIMENT_SCOPE="cost";CHANGED_DIMENSION="reporting_behavior_missingness_features"
FEATURES=["exp56_report_gap_days","exp56_history_count","exp56_missing_field_count","exp56_cost_unchanged_streak","exp56_spend_unchanged_streak","exp56_slippage_unchanged_streak","exp56_days_since_any_material_change"]
CRITICAL=["revised_cost_cr","cumulative_expenditure_cr","schedule_slippage_days","planned_completion_date","revised_completion_date"]
def engineer_reporting(frame:pd.DataFrame)->pd.DataFrame:
 out=frame.copy();out["snapshot_date"]=pd.to_datetime(out["snapshot_date"],errors="coerce")
 for f in FEATURES:out[f]=0.0
 for _,g in out.sort_values(["canonical_project_id","snapshot_date"]).groupby("canonical_project_id",sort=False):
  prev_date=None;prev={};streak={"revised_cost_cr":0,"cumulative_expenditure_cr":0,"schedule_slippage_days":0};last_change=None
  for pos,(idx,row) in enumerate(g.iterrows()):
   dt=pd.Timestamp(row["snapshot_date"]);out.at[idx,"exp56_history_count"]=pos+1;out.at[idx,"exp56_report_gap_days"]=0 if prev_date is None else max(0,(dt-prev_date).days);out.at[idx,"exp56_missing_field_count"]=sum(pd.isna(row.get(c)) for c in CRITICAL);changed=False
   for c in streak:
    cur=pd.to_numeric(pd.Series([row.get(c)]),errors="coerce").iloc[0];old=prev.get(c,np.nan)
    if pd.notna(cur) and pd.notna(old) and abs(float(cur)-float(old))<=1e-9:streak[c]+=1
    else:
     if pd.notna(cur) and pd.notna(old) and abs(float(cur)-float(old))>1e-9:changed=True
     streak[c]=0
    prev[c]=cur
   if changed:last_change=dt
   out.at[idx,"exp56_cost_unchanged_streak"]=streak["revised_cost_cr"];out.at[idx,"exp56_spend_unchanged_streak"]=streak["cumulative_expenditure_cr"];out.at[idx,"exp56_slippage_unchanged_streak"]=streak["schedule_slippage_days"];out.at[idx,"exp56_days_since_any_material_change"]=0 if last_change is None else max(0,(dt-last_change).days);prev_date=dt
 return out
def fit_experiment(**kwargs):return fit_features(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,dimension=CHANGED_DIMENSION,scope="cost",engineer=engineer_reporting,cost_new=FEATURES,details={"target_or_future_fields_used":False,"critical_reporting_fields":CRITICAL},**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
