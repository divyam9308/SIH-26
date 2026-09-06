"""Exp133: earned-schedule/calendar-velocity residual challenger."""
from __future__ import annotations
import numpy as np,pandas as pd
from backend.app.ml.experiments.post_exp113_delay_common import prepare_context,production_oof,fit_residual,persist

def earned(frame):
 x=frame.copy();planned=pd.to_numeric(x.get('planned_duration_days'),errors='coerce');progress=pd.to_numeric(x.get('physical_progress'),errors='coerce').clip(0,100);elapsed=pd.to_numeric(x.get('elapsed_duration_days'),errors='coerce')
 if elapsed.isna().all() and 'snapshot_date' in x and 'start_date' in x: elapsed=(pd.to_datetime(x.snapshot_date,errors='coerce')-pd.to_datetime(x.start_date,errors='coerce')).dt.days
 elapsed=elapsed.clip(lower=0);es=planned*progress/100.;v=es/elapsed.clip(lower=30);v=v.replace([np.inf,-np.inf],np.nan)
 # Conservative prior of planned velocity 1.0; elapsed-time shrinkage prevents early-stage explosions.
 k=180.;vs=(elapsed.fillna(0)*v.fillna(1.0)+k)/(elapsed.fillna(0)+k);vs=vs.clip(lower=.05,upper=2.0)
 rem=(planned-es).clip(lower=0)/vs;delay_es=(elapsed+rem-planned).clip(lower=0);prod=pd.to_numeric(x.get('production_prediction'),errors='coerce').fillna(0)
 x['es_velocity_shrunk']=vs.fillna(1.0);x['es_projected_delay_days']=delay_es.fillna(prod);x['es_divergence_gap']=(x.es_projected_delay_days-prod).clip(-1000,5000).fillna(0);return x

def fit_experiment(end=2021,output='reports/experiments/exp133_delay_2001_2021.json'):
 if end!=2021: raise ValueError('Exp133 audit is restricted to 2001-2021')
 ctx=prepare_context(end);oof=earned(production_oof(ctx,max_folds=6));score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];score=earned(score)
 features=['production_prediction','es_divergence_gap','es_velocity_shrunk','es_projected_delay_days','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage']
 corr,details=fit_residual(oof,score,features,133);gap=pd.to_numeric(score.es_divergence_gap,errors='coerce').fillna(0).to_numpy(float);base=float(details['cap']);upper=np.minimum(3000.,base+.5*np.maximum(gap,0));corr=np.minimum(corr,upper);pred=np.maximum(0,ctx['production_delay']+corr);details.update({'earned_schedule':'elapsed-time-shrunk calendar velocity','adaptive_upper_cap':True});return persist('exp133','Earned Schedule Velocity & Calendar Extension Hybrid',ctx,pred,details,output)
