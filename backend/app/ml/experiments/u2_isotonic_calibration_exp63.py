"""Experiment 63 / U2: lifecycle-aware isotonic calibration on Exp61 OOF predictions."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import cost_oof_frame, delay_oof_frame, production_comparison, run_cli

EXPERIMENT_ID="exp_63"; EXPERIMENT_SEQUENCE=63; MARKER="EXP63"
EXPERIMENT_NAME="U2 lifecycle-aware isotonic calibration on Exp61"; EXPERIMENT_SCOPE="cost+delay"
CHANGED_DIMENSION="global_plus_stage_shrunk_isotonic_mapping"
STRENGTH=80.0


def _fit_iso(oof: pd.DataFrame, score: pd.DataFrame, actual: str, nonnegative: bool=False):
    x=pd.to_numeric(oof["production_prediction"],errors="coerce").to_numpy(float)
    y=pd.to_numeric(oof[actual],errors="coerce").to_numpy(float)
    w=pd.to_numeric(oof["sample_weight"],errors="coerce").fillna(0).to_numpy(float)
    mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>=0)
    global_iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=0.0 if nonnegative else None)
    global_iso.fit(x[mask],y[mask],sample_weight=w[mask])
    sx=pd.to_numeric(score["production_prediction"],errors="coerce").to_numpy(float)
    out=np.asarray(global_iso.predict(sx),float)
    stage_details={}
    if "lifecycle_stage" in oof and "lifecycle_stage" in score:
        score_stage=score["lifecycle_stage"].astype("string")
        for stage,part in oof.loc[mask].groupby("lifecycle_stage",dropna=False):
            key="<NA>" if pd.isna(stage) else str(stage)
            px=pd.to_numeric(part["production_prediction"],errors="coerce").to_numpy(float)
            py=pd.to_numeric(part[actual],errors="coerce").to_numpy(float)
            pw=pd.to_numeric(part["sample_weight"],errors="coerce").fillna(0).to_numpy(float)
            good=np.isfinite(px)&np.isfinite(py)&np.isfinite(pw)&(pw>=0)
            support=float(pw[good].sum())
            if good.sum()<20 or len(np.unique(px[good]))<3 or support<=0:
                continue
            local=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=0.0 if nonnegative else None)
            local.fit(px[good],py[good],sample_weight=pw[good])
            if pd.isna(stage):
                smask=score_stage.isna().to_numpy(dtype=bool)
            else:
                smask=score_stage.eq(str(stage)).fillna(False).to_numpy(dtype=bool)
            if bool(smask.any()):
                alpha=support/(support+STRENGTH)
                lp=np.asarray(local.predict(sx[smask]),float)
                out[smask]=alpha*lp+(1-alpha)*out[smask]
                stage_details[key]={"support":support,"shrinkage_alpha":alpha,"rows":int(good.sum())}
    if nonnegative: out=np.maximum(0.0,out)
    return out,{"oof_rows":int(mask.sum()),"stage_maps":stage_details,"shrinkage_strength":STRENGTH,"holdout_tuned":False}


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,cohort,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end)
    coof=cost_oof_frame(data,production_bundle,training_start,training_end,test_end); cs=cohort.copy();cs["production_prediction"]=pc
    ec,cd=_fit_iso(coof,cs,"actual_cost_overrun_percentage",False)
    doof=delay_oof_frame(data,production_bundle,training_start,training_end,test_end); ds=cohort.copy();ds["production_prediction"]=pdly
    ed,dd=_fit_iso(doof,ds,"actual_delay_days",True)
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,cohort,pc,ec,pdly,ed,{"baseline":"current production (Exp61)","cost":cd,"delay":dd})

if __name__=="__main__": run_cli(sys.modules[__name__])
