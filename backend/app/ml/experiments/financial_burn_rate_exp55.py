"""Experiment 55 / D6: financial burn-rate completion features."""
from __future__ import annotations
import math,sys
import numpy as np
import pandas as pd
from backend.app.ml.experiments.nextgen_common import fit_features,run_cli
EXPERIMENT_ID="exp_55"; EXPERIMENT_SEQUENCE=55; MARKER="EXP55"
EXPERIMENT_NAME="Financial burn-rate Delay specialist features"; EXPERIMENT_SCOPE="delay"
HYPOTHESIS="As-of financial execution velocity and implied months-to-spend-remaining provide independent completion-time signal."
CHANGED_DIMENSION="financial_burn_rate_completion_features"
FEATURES=["exp55_spend_velocity_3m","exp55_spend_velocity_6m","exp55_spend_acceleration","exp55_remaining_revised_spend_pct","exp55_implied_months_to_revised_spend","exp55_burn_vs_duration_gap"]
def _slope(dates,vals,current,days):
 idx=[i for i,d in enumerate(dates) if 0<=(current-d).days<=days and np.isfinite(vals[i])]
 if len(idx)<2:return 0.0
 x=np.asarray([(dates[i]-dates[idx[0]]).days/30.4375 for i in idx],float);y=np.asarray([vals[i] for i in idx],float)
 if np.ptp(x)<=1e-12:return 0.0
 x=x-x.mean();return float(np.dot(x,y-y.mean())/np.dot(x,x))
def engineer_burn_rate(frame:pd.DataFrame)->pd.DataFrame:
 out=frame.copy();[out.__setitem__(f,0.0) for f in FEATURES];out["snapshot_date"]=pd.to_datetime(out["snapshot_date"],errors="coerce")
 for _,g in out.sort_values(["canonical_project_id","snapshot_date"]).groupby("canonical_project_id",sort=False):
  dates=[];ratios=[]
  for idx,row in g.iterrows():
   dates.append(pd.Timestamp(row["snapshot_date"]));spend=pd.to_numeric(pd.Series([row.get("cumulative_expenditure_cr")]),errors="coerce").iloc[0];revised=pd.to_numeric(pd.Series([row.get("revised_cost_cr")]),errors="coerce").iloc[0];ratio=float(spend/revised*100) if pd.notna(spend) and pd.notna(revised) and revised>0 else math.nan;ratios.append(ratio);v3=_slope(dates,ratios,dates[-1],92);v6=_slope(dates,ratios,dates[-1],183);remaining=max(0.0,100.0-ratio) if np.isfinite(ratio) else 100.0;implied=remaining/max(v3,0.05);duration=float(row.get("duration_ratio") or 0.0) if pd.notna(row.get("duration_ratio")) else 0.0
   out.at[idx,"exp55_spend_velocity_3m"]=v3;out.at[idx,"exp55_spend_velocity_6m"]=v6;out.at[idx,"exp55_spend_acceleration"]=v3-v6;out.at[idx,"exp55_remaining_revised_spend_pct"]=remaining;out.at[idx,"exp55_implied_months_to_revised_spend"]=min(implied,240.0);out.at[idx,"exp55_burn_vs_duration_gap"]=min(implied,240.0)-duration*12.0
 return out
def fit_experiment(**kwargs):return fit_features(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,dimension=CHANGED_DIMENSION,scope="delay",engineer=engineer_burn_rate,delay_new=FEATURES,details={"velocity_windows_months":[3,6],"minimum_velocity_floor_pct_per_month":0.05},**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
