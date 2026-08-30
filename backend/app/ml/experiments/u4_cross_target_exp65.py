"""Experiment 65 / U4: leakage-safe cross-target OOF residual stacking."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import cost_oof_frame,delay_oof_frame,production_comparison,run_cli,weighted_quantile
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel

EXPERIMENT_ID="exp_65";EXPERIMENT_SEQUENCE=65;MARKER="EXP65"
EXPERIMENT_NAME="U4 cross-target OOF Cost-Delay stacking";EXPERIMENT_SCOPE="cost+delay"
CHANGED_DIMENSION="opposite_target_oof_prediction_as_residual_meta_feature"
BASE=["cost_prediction","delay_prediction","cost_escalation_percentage","schedule_slippage_days","duration_ratio","expenditure_ratio","progress_deviation","approved_cost_cr"]


def _aligned(cost,delay):
    keys=["canonical_project_id","snapshot_date"]
    c=cost.copy().rename(columns={"production_prediction":"cost_prediction","residual":"cost_residual"})
    d=delay[keys+["production_prediction","residual"]].copy().rename(columns={"production_prediction":"delay_prediction","residual":"delay_residual"})
    return c.merge(d,on=keys,how="inner",validate="one_to_one")


def _fit_residual(train,score,target,seed):
    cols=[c for c in BASE if c in train and c in score]
    med={c:float(pd.to_numeric(train[c],errors="coerce").median()) for c in cols}
    x=pd.DataFrame({c:pd.to_numeric(train[c],errors="coerce").fillna(med[c]) for c in cols});xs=pd.DataFrame({c:pd.to_numeric(score[c],errors="coerce").fillna(med[c]) for c in cols})
    y=pd.to_numeric(train[target],errors="coerce").fillna(0).to_numpy(float);w=pd.to_numeric(train["sample_weight"],errors="coerce").fillna(0).to_numpy(float)
    model=LGBMRegressor(n_estimators=170,learning_rate=.025,max_depth=3,num_leaves=12,min_child_samples=70,reg_alpha=6,reg_lambda=30,random_state=seed,verbosity=-1)
    model.fit(x,y,sample_weight=w);corr=np.asarray(model.predict(xs),float);cap=weighted_quantile(np.abs(y),w,.90);return np.clip(corr,-cap,cap),{"features":cols,"oof_rows":len(train),"correction_cap_q90":cap}


def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,c,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end)
    joined=_aligned(cost_oof_frame(data,production_bundle,training_start,training_end,test_end),delay_oof_frame(data,production_bundle,training_start,training_end,test_end))
    score=c.copy();score["cost_prediction"]=pc;score["delay_prediction"]=pdly
    elig=AFTResidualDelayModel._aft_eligible(score).to_numpy(bool);ec=pc.copy();ed=pdly.copy();details={"aligned_oof_rows":len(joined),"eligible_score_rows":int(elig.sum()),"fallback_rows_retained":int((~elig).sum())}
    if elig.any():
        sub=score.loc[elig].copy();cc,cd=_fit_residual(joined,sub,"cost_residual",6501);dc,dd=_fit_residual(joined,sub,"delay_residual",6502)
        ec[elig]=pc[elig]+cc;ed[elig]=np.maximum(0,pdly[elig]+dc);details.update({"cost":cd,"delay":dd})
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,ec,pdly,ed,{"baseline":"assumed Exp61 production from PR #96",**details,"holdout_used_for_fit":False})

if __name__=="__main__":run_cli(sys.modules[__name__])
