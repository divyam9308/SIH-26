"""Experiment 114: earned-schedule structural signals beyond current U1 Delay."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_114';EXPERIMENT_NAME='Earned-schedule completion forecast';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=114
EARNED_FEATURES=['exp114_progress_fraction','exp114_time_fraction','exp114_schedule_performance_index','exp114_implied_total_duration','exp114_implied_remaining_days','exp114_implied_delay','exp114_structural_gap']

def _engineer(frame):
    out=frame.copy();progress=pd.to_numeric(out.get('physical_progress'),errors='coerce')/100.0;progress=progress.clip(.01,1.0);elapsed=pd.to_numeric(out.get('elapsed_duration_days'),errors='coerce');planned=pd.to_numeric(out.get('planned_duration_days'),errors='coerce');time_fraction=(elapsed/planned.replace(0,np.nan)).clip(lower=0,upper=5);spi=(progress/time_fraction.replace(0,np.nan)).clip(lower=.05,upper=5);implied_total=(elapsed/progress).clip(lower=0,upper=10000);remaining=(implied_total-elapsed).clip(lower=0,upper=6000);snapshot=pd.to_datetime(out.get('snapshot_date'),errors='coerce');planned_date=pd.to_datetime(out.get('planned_completion_date'),errors='coerce');completion=snapshot+pd.to_timedelta(remaining.fillna(0),unit='D');implied_delay=((completion-planned_date).dt.total_seconds()/86400.0).clip(lower=0,upper=6000)
    out['exp114_progress_fraction']=progress;out['exp114_time_fraction']=time_fraction;out['exp114_schedule_performance_index']=spi;out['exp114_implied_total_duration']=implied_total;out['exp114_implied_remaining_days']=remaining;out['exp114_implied_delay']=implied_delay;out['exp114_structural_gap']=implied_delay-pd.to_numeric(out.get('schedule_slippage_days'),errors='coerce');return out

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;features=['production_prediction','u1_correction']+EARNED_FEATURES+['duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','progress_deviation','exp58_group_support'];corr,details=fit_residual_booster(oof,score,features,11401);details.update({'earned_schedule_features':EARNED_FEATURES,'feature_semantics':'deterministic as-of project-management structural estimates','holdout_target_used_in_engineering':False});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
