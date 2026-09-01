"""Experiment 116: structural agency/sector queue-congestion priors beyond U1."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_116';EXPERIMENT_NAME='Queueing-based execution congestion prior';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=116
QUEUE_FEATURES=['exp116_agency_queue_pressure','exp116_sector_queue_pressure','exp116_agency_wait_proxy_days','exp116_sector_wait_proxy_days']

def _group_queue(out,key,prefix):
    month=out['_exp116_month'];temp=pd.DataFrame({'key':out[key].astype('string').fillna('<NA>'),'month':month,'project':out['canonical_project_id']})
    active=temp.groupby(['key','month'])['project'].nunique().rename('active').reset_index().sort_values(['key','month'])
    events=pd.DataFrame({'project':out['canonical_project_id'],'key':out[key].astype('string').fillna('<NA>'),'completion':pd.to_datetime(out['completion_date'],errors='coerce')}).dropna(subset=['completion']).sort_values('completion').drop_duplicates('project',keep='first');events['month']=events['completion'].dt.to_period('M').dt.to_timestamp();comp=events.groupby(['key','month']).size().rename('completions').reset_index();active=active.merge(comp,on=['key','month'],how='left');active['completions']=active['completions'].fillna(0.0);g=active.groupby('key',sort=False);active['lag_active']=g['active'].shift(1);active['throughput']=g['completions'].transform(lambda s:s.shift(1).rolling(12,min_periods=1).sum());active[f'{prefix}_queue_pressure']=active['lag_active']/(active['throughput']+1.0);return active[['key','month',f'{prefix}_queue_pressure']]

def _engineer(frame):
    out=frame.copy();out['_exp116_month']=pd.to_datetime(out['snapshot_date'],errors='coerce').dt.to_period('M').dt.to_timestamp();agency=_group_queue(out,'implementing_agency','exp116_agency');sector=_group_queue(out,'sector','exp116_sector');out['_exp116_agency']=out.get('implementing_agency',pd.Series('<NA>',index=out.index)).astype('string').fillna('<NA>');out['_exp116_sector']=out.get('sector',pd.Series('<NA>',index=out.index)).astype('string').fillna('<NA>');agency=agency.rename(columns={'key':'_exp116_agency','month':'_exp116_month'});sector=sector.rename(columns={'key':'_exp116_sector','month':'_exp116_month'});out=out.merge(agency,on=['_exp116_agency','_exp116_month'],how='left',sort=False).merge(sector,on=['_exp116_sector','_exp116_month'],how='left',sort=False);planned=pd.to_numeric(out.get('planned_duration_days'),errors='coerce').clip(lower=30,upper=5000);out['exp116_agency_wait_proxy_days']=(out['exp116_agency_queue_pressure']*planned/12.0).clip(0,3000);out['exp116_sector_wait_proxy_days']=(out['exp116_sector_queue_pressure']*planned/12.0).clip(0,3000);return out.drop(columns=['_exp116_month','_exp116_agency','_exp116_sector'])

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;features=['production_prediction','u1_correction']+QUEUE_FEATURES+['duration_ratio','schedule_slippage_days','expenditure_ratio','progress_deviation','exp58_group_support'];corr,details=fit_residual_booster(oof,score,features,11601);details.update({'queue_features':QUEUE_FEATURES,'queue_load_lag_months':1,'throughput_window_report_months':12,'same_or_future_completion_events_used':False});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
