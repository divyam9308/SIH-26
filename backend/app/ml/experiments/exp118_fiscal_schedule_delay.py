"""Experiment 118: fiscal funding-to-schedule coupling beyond current U1 Delay."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_118';EXPERIMENT_NAME='Fiscal funding-to-schedule coupling';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=118
FISCAL_FEATURES=['exp118_fiscal_month','exp118_fiscal_q4','exp118_spend_velocity','exp118_spend_acceleration','exp118_spend_slowdown','exp118_q4_spend_velocity','exp118_slowdown_slippage','exp118_slowdown_progress_gap']

def _engineer(frame):
    out=frame.copy();out['_exp118_date']=pd.to_datetime(out['snapshot_date'],errors='coerce');out=out.sort_values(['canonical_project_id','_exp118_date']).copy();month=out['_exp118_date'].dt.month;fiscal_month=((month-4)%12)+1;out['exp118_fiscal_month']=fiscal_month.astype(float);out['exp118_fiscal_q4']=(fiscal_month>=10).astype(float);spend=pd.to_numeric(out.get('expenditure_ratio'),errors='coerce');g=out.groupby('canonical_project_id',sort=False);days=g['_exp118_date'].diff().dt.total_seconds()/86400.0;months=(days/30.4375).replace(0,np.nan);velocity=g['expenditure_ratio'].diff()/months;out['exp118_spend_velocity']=pd.to_numeric(velocity,errors='coerce');out['exp118_spend_acceleration']=out.groupby('canonical_project_id',sort=False)['exp118_spend_velocity'].diff();out['exp118_spend_slowdown']=(-out['exp118_spend_acceleration']).clip(lower=0);out['exp118_q4_spend_velocity']=out['exp118_fiscal_q4']*out['exp118_spend_velocity'];slip=pd.to_numeric(out.get('schedule_slippage_days'),errors='coerce');gap=pd.to_numeric(out.get('progress_deviation'),errors='coerce');out['exp118_slowdown_slippage']=out['exp118_spend_slowdown']*slip;out['exp118_slowdown_progress_gap']=out['exp118_spend_slowdown']*np.abs(gap);return out.drop(columns=['_exp118_date'])

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;features=['production_prediction','u1_correction']+FISCAL_FEATURES+['duration_ratio','schedule_slippage_days','expenditure_ratio','progress_deviation','cost_escalation_percentage','exp58_group_support'];corr,details=fit_residual_booster(oof,score,features,11801);details.update({'fiscal_features':FISCAL_FEATURES,'fiscal_year_start_month':4,'trajectory_features_use_current_and_prior_reports_only':True});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
