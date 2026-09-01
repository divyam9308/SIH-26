"""Experiment 115: lagged agency backlog and throughput signals beyond U1."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_115';EXPERIMENT_NAME='Dynamic agency backlog and throughput index';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=115
PORTFOLIO_FEATURES=['exp115_lag_active_projects','exp115_active_6m_mean','exp115_backlog_change','exp115_completions_12m','exp115_backlog_throughput_pressure']

def _engineer(frame):
    out=frame.copy();out['_exp115_date']=pd.to_datetime(out['snapshot_date'],errors='coerce');out['_exp115_month']=out['_exp115_date'].dt.to_period('M').dt.to_timestamp();out['_exp115_agency']=out.get('implementing_agency',pd.Series('<NA>',index=out.index)).astype('string').fillna('<NA>')
    monthly=out.groupby(['_exp115_agency','_exp115_month'],dropna=False)['canonical_project_id'].nunique().rename('active').reset_index().sort_values(['_exp115_agency','_exp115_month'])
    events=out[['canonical_project_id','_exp115_agency','completion_date']].copy();events['completion_date']=pd.to_datetime(events['completion_date'],errors='coerce');events=events.dropna(subset=['completion_date']).sort_values('completion_date').drop_duplicates('canonical_project_id',keep='first');events['_exp115_month']=events['completion_date'].dt.to_period('M').dt.to_timestamp();counts=events.groupby(['_exp115_agency','_exp115_month']).size().rename('completions').reset_index();monthly=monthly.merge(counts,on=['_exp115_agency','_exp115_month'],how='left');monthly['completions']=monthly['completions'].fillna(0.0)
    g=monthly.groupby('_exp115_agency',sort=False);monthly['exp115_lag_active_projects']=g['active'].shift(1);monthly['exp115_active_6m_mean']=g['active'].transform(lambda s:s.shift(1).rolling(6,min_periods=1).mean());monthly['exp115_completions_12m']=g['completions'].transform(lambda s:s.shift(1).rolling(12,min_periods=1).sum());monthly['exp115_backlog_change']=monthly['exp115_lag_active_projects']-monthly['exp115_active_6m_mean'];monthly['exp115_backlog_throughput_pressure']=monthly['exp115_lag_active_projects']/(monthly['exp115_completions_12m']+1.0)
    out=out.merge(monthly[['_exp115_agency','_exp115_month']+PORTFOLIO_FEATURES],on=['_exp115_agency','_exp115_month'],how='left',sort=False);return out.drop(columns=['_exp115_date','_exp115_month','_exp115_agency'])

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;features=['production_prediction','u1_correction']+PORTFOLIO_FEATURES+['duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','exp58_group_support'];corr,details=fit_residual_booster(oof,score,features,11501);details.update({'portfolio_features':PORTFOLIO_FEATURES,'same_month_portfolio_signal_used':False,'completion_counts_are_lagged':True});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
