"""Experiment 71 / U10: robust multi-resolution Cost trajectory basis."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp35_aft_residual_combo import _corrections
from backend.app.ml.experiments.nextgen_common import _compare,_cost_oof,_family,_persist,_prepare,normalize_taxonomy,shrunk_calibration
from backend.app.ml.experiments.post61_common import run_cli
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED

EXPERIMENT_ID="exp_71";EXPERIMENT_SEQUENCE=71;MARKER="EXP71"
EXPERIMENT_NAME="U10 robust multi-resolution Cost trajectory basis";EXPERIMENT_SCOPE="cost"
CHANGED_DIMENSION="prefix_theil_sen_slope_components_and_curvature"
WINDOWS=(3,6,12,24,36);SIGNALS=("cost_escalation_percentage","expenditure_ratio")
TREND_FEATURES=[f"exp71_{s}_slope_{w}m" for s in SIGNALS for w in WINDOWS]+[f"exp71_{s}_curvature_{a}_{b}" for s in SIGNALS for a,b in ((3,12),(6,24),(12,36))]


def _median_pairwise_slope(days,values):
    x=np.asarray(days,float);y=np.asarray(values,float);m=np.isfinite(x)&np.isfinite(y);x=x[m];y=y[m]
    if len(x)<2:return np.nan
    i,j=np.triu_indices(len(x),1);dx=x[j]-x[i];good=dx>0
    if not good.any():return np.nan
    return float(np.median((y[j][good]-y[i][good])/dx[good]*30.4375))


def enrich_robust_trends(frame: pd.DataFrame) -> pd.DataFrame:
    out=frame.copy();out["snapshot_date"]=pd.to_datetime(out["snapshot_date"],errors="coerce");result={name:pd.Series(np.nan,index=out.index,dtype=float) for name in TREND_FEATURES}
    ordered=out.sort_values(["canonical_project_id","snapshot_date"],kind="mergesort")
    for _,part in ordered.groupby("canonical_project_id",sort=False):
        idx=part.index.to_numpy();days=part["snapshot_date"].astype("int64").to_numpy(float)/86400e9
        for signal in SIGNALS:
            values=pd.to_numeric(part.get(signal),errors="coerce").to_numpy(float)
            slopes={w:np.full(len(part),np.nan,float) for w in WINDOWS}
            for pos in range(len(part)):
                for w in WINDOWS:
                    lo=int(np.searchsorted(days,days[pos]-w*30.4375,side="left"));slopes[w][pos]=_median_pairwise_slope(days[lo:pos+1],values[lo:pos+1])
            for w in WINDOWS: result[f"exp71_{signal}_slope_{w}m"].loc[idx]=slopes[w]
            for a,b in ((3,12),(6,24),(12,36)): result[f"exp71_{signal}_curvature_{a}_{b}"].loc[idx]=slopes[a]-slopes[b]
    for name,series in result.items():out[name]=series
    return out


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    frame=enrich_robust_trends(normalize_taxonomy(_prepare(data)));train,test=temporal_project_split(frame,training_start,training_end,test_end);c=_compare(test);cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c),float);pdly=np.maximum(0,np.asarray(dm.predict(c),float))
    features=list(dict.fromkeys(list(cm.features)+TREND_FEATURES));fam=_family(cm);cal=shrunk_calibration(_cost_oof(train,features,fam),40.0);model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],train,features,"actual_cost_overrun_percentage");raw=np.asarray(model.predict(c[features]),float);ec=raw+_corrections(c,raw,cal)
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,ec,pdly,pdly.copy(),{"baseline":"assumed Exp61 production from PR #96","trend_features":TREND_FEATURES,"windows_months":list(WINDOWS),"slope_method":"median of all pairwise prefix slopes (Theil-Sen slope component)","future_reports_used":False,"delay_predictions_identical":True})

if __name__=="__main__":run_cli(sys.modules[__name__])
