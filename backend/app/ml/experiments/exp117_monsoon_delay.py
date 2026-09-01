"""Experiment 117: calendar monsoon-exposure proxy beyond current U1 Delay.

No tracked historical IMD/weather series exists in this repository. This
experiment therefore tests only deterministic calendar exposure and explicitly
does not claim observed rainfall, flood or cyclone information.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract

EXPERIMENT_ID='exp_117';EXPERIMENT_NAME='Calendar monsoon-exposure Delay proxy';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=117
MONSOON_FEATURES=['exp117_monsoon','exp117_month_sin','exp117_month_cos','exp117_civil_exposure','exp117_monsoon_slippage','exp117_monsoon_progress_gap','exp117_civil_monsoon_stress']

def _engineer(frame):
    out=frame.copy();dt=pd.to_datetime(out['snapshot_date'],errors='coerce');month=dt.dt.month.astype(float);monsoon=month.isin([6,7,8,9]).astype(float);sector=out.get('sector',pd.Series('',index=out.index)).astype('string').fillna('').str.lower();civil=sector.str.contains('road|highway|rail|transport|bridge|port|irrigation|water|construction|urban',regex=True).astype(float);slip=pd.to_numeric(out.get('schedule_slippage_days'),errors='coerce');gap=pd.to_numeric(out.get('progress_deviation'),errors='coerce')
    out['exp117_monsoon']=monsoon;out['exp117_month_sin']=np.sin(2*np.pi*month/12.0);out['exp117_month_cos']=np.cos(2*np.pi*month/12.0);out['exp117_civil_exposure']=civil;out['exp117_monsoon_slippage']=monsoon*slip;out['exp117_monsoon_progress_gap']=monsoon*gap;out['exp117_civil_monsoon_stress']=civil*monsoon*(slip.fillna(0)+np.abs(gap.fillna(0))*10.0);return out

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;features=['production_prediction','u1_correction']+MONSOON_FEATURES+['duration_ratio','schedule_slippage_days','expenditure_ratio','progress_deviation','exp58_group_support'];corr,details=fit_residual_booster(oof,score,features,11701);details.update({'weather_data_source':'none','observed_weather_used':False,'proxy':'calendar Jun-Sep monsoon exposure plus sector interactions','monsoon_features':MONSOON_FEATURES});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
