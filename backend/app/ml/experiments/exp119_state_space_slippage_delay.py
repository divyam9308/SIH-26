"""Experiment 119: causal state-space schedule-slippage features beyond U1."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_119';EXPERIMENT_NAME='State-space schedule-slippage forecast';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=119
STATE_FEATURES=['exp119_slip_level','exp119_slip_trend','exp119_slip_innovation','exp119_slip_variance','exp119_slip_projection_3','exp119_slip_projection_6']
LEVEL_GAIN=.35;TREND_GAIN=.12;VAR_GAIN=.20

def _engineer(frame):
    out=frame.copy();out['_exp119_date']=pd.to_datetime(out['snapshot_date'],errors='coerce');out=out.sort_values(['canonical_project_id','_exp119_date']).copy()
    for c in STATE_FEATURES: out[c]=np.nan
    for _,g in out.groupby('canonical_project_id',sort=False):
        level=None;trend=0.0;variance=0.0
        for idx in g.index:
            value=pd.to_numeric(pd.Series([out.at[idx,'schedule_slippage_days']]),errors='coerce').iloc[0]
            if pd.isna(value): continue
            value=float(value)
            if level is None:
                level=value;innovation=0.0;trend=0.0;variance=0.0
            else:
                innovation=value-level;old_level=level;level=level+LEVEL_GAIN*innovation;trend=(1-TREND_GAIN)*trend+TREND_GAIN*(level-old_level);variance=(1-VAR_GAIN)*variance+VAR_GAIN*(innovation**2)
            out.at[idx,'exp119_slip_level']=level;out.at[idx,'exp119_slip_trend']=trend;out.at[idx,'exp119_slip_innovation']=innovation;out.at[idx,'exp119_slip_variance']=variance;out.at[idx,'exp119_slip_projection_3']=max(0.0,level+3.0*trend);out.at[idx,'exp119_slip_projection_6']=max(0.0,level+6.0*trend)
    return out.drop(columns=['_exp119_date'])

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;features=['production_prediction','u1_correction']+STATE_FEATURES+['duration_ratio','schedule_slippage_days','expenditure_ratio','progress_deviation','cost_escalation_percentage','exp58_group_support'];corr,details=fit_residual_booster(oof,score,features,11901);details.update({'state_features':STATE_FEATURES,'level_gain':LEVEL_GAIN,'trend_gain':TREND_GAIN,'variance_gain':VAR_GAIN,'filter_is_causal_prefix_only':True});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
