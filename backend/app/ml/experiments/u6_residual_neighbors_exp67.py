"""Experiment 67 / U6: historical-neighbor correction of Exp61 OOF residuals."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import cost_oof_frame,delay_oof_frame,production_comparison,run_cli,weighted_quantile

EXPERIMENT_ID="exp_67";EXPERIMENT_SEQUENCE=67;MARKER="EXP67"
EXPERIMENT_NAME="U6 residual-neighbor correction on Exp61";EXPERIMENT_SCOPE="cost+delay"
CHANGED_DIMENSION="nearest_historical_oof_residual_median"
FEATURES=["production_prediction","cost_escalation_percentage","schedule_slippage_days","duration_ratio","expenditure_ratio","progress_deviation","approved_cost_cr","exp58_delay_hier_prior","exp58_group_support"]
K=40


def _neighbor_correction(oof,score):
    cols=[c for c in FEATURES if c in oof and c in score]
    med={c:float(pd.to_numeric(oof[c],errors="coerce").median()) for c in cols}
    x=np.column_stack([pd.to_numeric(oof[c],errors="coerce").fillna(med[c]).to_numpy(float) for c in cols]);xs=np.column_stack([pd.to_numeric(score[c],errors="coerce").fillna(med[c]).to_numpy(float) for c in cols])
    scaler=StandardScaler().fit(x);xt=scaler.transform(x);xst=scaler.transform(xs);k=min(K,len(oof));nn=NearestNeighbors(n_neighbors=k,metric="euclidean").fit(xt);dist,idx=nn.kneighbors(xst)
    residual=pd.to_numeric(oof["residual"],errors="coerce").fillna(0).to_numpy(float);base_w=pd.to_numeric(oof["sample_weight"],errors="coerce").fillna(0).to_numpy(float);corr=np.zeros(len(score),float)
    for i in range(len(score)):
        local=idx[i];w=base_w[local]/np.maximum(dist[i],1e-3);corr[i]=weighted_quantile(residual[local],w,.5)
    cap=weighted_quantile(np.abs(residual),base_w,.90);corr=np.clip(corr,-cap,cap)
    return corr,{"features":cols,"neighbors":k,"oof_rows":len(oof),"correction_cap_q90":cap,"predicts_outcome_directly":False}


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,c,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end)
    co=cost_oof_frame(data,production_bundle,training_start,training_end,test_end);cs=c.copy();cs["production_prediction"]=pc;cc,cd=_neighbor_correction(co,cs)
    do=delay_oof_frame(data,production_bundle,training_start,training_end,test_end);ds=c.copy();ds["production_prediction"]=pdly;dc,dd=_neighbor_correction(do,ds)
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,pc+cc,pdly,np.maximum(0,pdly+dc),{"baseline":"assumed Exp61 production from PR #96","cost":cd,"delay":dd,"holdout_used_for_neighbor_fit":False})

if __name__=="__main__":run_cli(sys.modules[__name__])
