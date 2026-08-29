"""Experiment 60 / C10+D10: data-quality gating + missing-planned fallback specialist."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from backend.app.ml.experiments.nextgen_common import fit_quality,run_cli
EXPERIMENT_ID="exp_60";EXPERIMENT_SEQUENCE=60;MARKER="EXP60";EXPERIMENT_NAME="Data-quality gating and missing-planned fallback specialist";EXPERIMENT_SCOPE="cost+delay";CHANGED_DIMENSION="data_quality_features_plus_missing_planned_fallback_specialist"
FEATURES=["exp60_missing_count","exp60_duration_implausible","exp60_cost_implausible","exp60_spend_implausible","exp60_approval_date_future","exp60_planned_date_missing","exp60_quality_score"]
def engineer_quality(frame:pd.DataFrame)->pd.DataFrame:
 out=frame.copy();duration=pd.to_numeric(out.get("duration_ratio"),errors="coerce");cost=pd.to_numeric(out.get("cost_escalation_percentage"),errors="coerce");spend=pd.to_numeric(out.get("expenditure_ratio"),errors="coerce");snapshot=pd.to_datetime(out.get("snapshot_date"),errors="coerce");approval=pd.to_datetime(out.get("approval_date"),errors="coerce") if "approval_date" in out else pd.Series(pd.NaT,index=out.index);planned=pd.to_datetime(out.get("planned_completion_date"),errors="coerce");critical=["approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","schedule_slippage_days","duration_ratio","planned_completion_date"]
 out["exp60_missing_count"]=sum(out.get(c,pd.Series(np.nan,index=out.index)).isna().astype(int) for c in critical).astype(float);out["exp60_duration_implausible"]=((duration<0)|(duration>10)).fillna(False).astype(float);out["exp60_cost_implausible"]=((cost<-100)|(cost>1000)).fillna(False).astype(float);out["exp60_spend_implausible"]=((spend<0)|(spend>3)).fillna(False).astype(float);out["exp60_approval_date_future"]=(approval>snapshot).fillna(False).astype(float);out["exp60_planned_date_missing"]=planned.isna().astype(float);penalty=out[["exp60_missing_count","exp60_duration_implausible","exp60_cost_implausible","exp60_spend_implausible","exp60_approval_date_future","exp60_planned_date_missing"]].sum(axis=1);out["exp60_quality_score"]=1.0/(1.0+penalty);return out
def fit_experiment(**kwargs):return fit_quality(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,engineer=engineer_quality,features=FEATURES,**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
