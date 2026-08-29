"""Experiment 57 / D5: planned-date reliability evidence router."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from backend.app.ml.experiments.nextgen_common import fit_router,run_cli
EXPERIMENT_ID="exp_57";EXPERIMENT_SEQUENCE=57;MARKER="EXP57";EXPERIMENT_NAME="Planned-date reliability Delay router";EXPERIMENT_SCOPE="delay";CHANGED_DIMENSION="planned_date_reliability_evidence_router"
SCORE="exp57_planned_date_reliability";FEATURES=[SCORE,"exp57_schedule_revision_count","exp57_schedule_revision_abs_mean_days","exp57_days_since_schedule_revision","exp57_effective_date_missing"]
def engineer_reliability(frame:pd.DataFrame)->pd.DataFrame:
 out=frame.copy();out["snapshot_date"]=pd.to_datetime(out["snapshot_date"],errors="coerce");planned=pd.to_datetime(out.get("planned_completion_date"),errors="coerce");revised=pd.to_datetime(out.get("revised_completion_date"),errors="coerce") if "revised_completion_date" in out else pd.Series(pd.NaT,index=out.index);out["_effective"]=revised.where(revised.notna(),planned)
 for f in FEATURES:out[f]=0.0
 for _,g in out.sort_values(["canonical_project_id","snapshot_date"]).groupby("canonical_project_id",sort=False):
  prev=None;moves=[];last_revision=None
  for idx,row in g.iterrows():
   cur=out.at[idx,"_effective"];dt=pd.Timestamp(row["snapshot_date"])
   if pd.notna(cur) and pd.notna(prev):
    move=float((pd.Timestamp(cur)-pd.Timestamp(prev)).days)
    if abs(move)>=14:moves.append(move);last_revision=dt
   missing=float(pd.isna(cur));count=len(moves);mean_abs=float(np.mean(np.abs(moves))) if moves else 0.0;recency=9999.0 if last_revision is None else float((dt-last_revision).days);reliability=float(np.exp(-mean_abs/365.0)/(1.0+0.15*count))*(0.0 if missing else 1.0)
   out.at[idx,SCORE]=reliability;out.at[idx,"exp57_schedule_revision_count"]=count;out.at[idx,"exp57_schedule_revision_abs_mean_days"]=mean_abs;out.at[idx,"exp57_days_since_schedule_revision"]=recency;out.at[idx,"exp57_effective_date_missing"]=missing;prev=cur
 return out.drop(columns="_effective")
def fit_experiment(**kwargs):return fit_router(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,engineer=engineer_reliability,score=SCORE,**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
