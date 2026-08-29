"""Experiment 53 / C5: same-date cohort-relative Cost pressure."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from backend.app.ml.experiments.nextgen_common import fit_features,run_cli
EXPERIMENT_ID="exp_53"; EXPERIMENT_SEQUENCE=53; MARKER="EXP53"
EXPERIMENT_NAME="Same-date peer pressure Cost features"; EXPERIMENT_SCOPE="cost"
HYPOTHESIS="A project's observable state relative to contemporaneous projects captures macro/sector pressure not represented by absolute trajectories."
CHANGED_DIMENSION="same_date_peer_relative_features"
FEATURES=["exp53_peer_count","exp53_cost_rank","exp53_spend_rank","exp53_slippage_rank","exp53_duration_rank","exp53_cost_sector_rank","exp53_slippage_sector_rank"]
def engineer_peer_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    out=frame.copy();out["_month"]=pd.to_datetime(out["snapshot_date"],errors="coerce").dt.to_period("M").astype("string");out["_sector"]=out.get("sector",pd.Series("<NA>",index=out.index)).astype("string").fillna("<NA>")
    numeric={"cost":pd.to_numeric(out.get("cost_escalation_percentage"),errors="coerce"),"spend":pd.to_numeric(out.get("expenditure_ratio"),errors="coerce"),"slippage":pd.to_numeric(out.get("schedule_slippage_days"),errors="coerce"),"duration":pd.to_numeric(out.get("duration_ratio"),errors="coerce")}
    out["exp53_peer_count"]=out.groupby("_month")["canonical_project_id"].transform("nunique").astype(float)
    for name,series in numeric.items():out[f"exp53_{name}_rank"]=series.groupby(out["_month"]).rank(pct=True,method="average").fillna(.5)
    out["exp53_cost_sector_rank"]=numeric["cost"].groupby([out["_month"],out["_sector"]]).rank(pct=True,method="average").fillna(.5);out["exp53_slippage_sector_rank"]=numeric["slippage"].groupby([out["_month"],out["_sector"]]).rank(pct=True,method="average").fillna(.5)
    return out.drop(columns=["_month","_sector"])
def fit_experiment(**kwargs):return fit_features(exp_id=EXPERIMENT_ID,name=EXPERIMENT_NAME,dimension=CHANGED_DIMENSION,scope="cost",engineer=engineer_peer_pressure,cost_new=FEATURES,details={"peer_definition":"same snapshot month; sector ranks use same month+sector; no targets"},**kwargs)
if __name__=="__main__":run_cli(sys.modules[__name__])
