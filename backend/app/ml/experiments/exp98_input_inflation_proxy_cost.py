"""Exp98: leakage-safe sector construction-input pressure proxy.

The repository has no tracked official WPI input-price history, so this experiment
intentionally does not fabricate one. It tests an internal as-of sector price-
pressure proxy from lagged cross-project Cost-revision movements.
"""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_98';EXPERIMENT_NAME='Construction-input inflation exposure proxy';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=98

def _engineer(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x['_month']=x['snapshot_date'].dt.to_period('M').dt.to_timestamp();x=x.sort_values(['canonical_project_id','snapshot_date']);esc=pd.to_numeric(x['cost_escalation_percentage'],errors='coerce');x['_esc_change']=esc.groupby(x['canonical_project_id']).diff();monthly=x.groupby(['sector','_month'],dropna=False).agg(_pressure=('_esc_change','median'),_dispersion=('_esc_change','std'),_projects=('canonical_project_id','nunique')).reset_index().sort_values(['sector','_month'])
    for win in (3,6,12):
        monthly[f'exp98_pressure_{win}m']=monthly.groupby('sector')['_pressure'].transform(lambda s:s.shift(1).rolling(win,min_periods=1).mean());monthly[f'exp98_dispersion_{win}m']=monthly.groupby('sector')['_dispersion'].transform(lambda s:s.shift(1).rolling(win,min_periods=1).mean())
    keep=['sector','_month']+[c for c in monthly.columns if c.startswith('exp98_')];x=x.merge(monthly[keep],on=['sector','_month'],how='left',sort=False);approved=pd.to_numeric(x['approved_cost_cr'],errors='coerce');x['exp98_size_exposure']=np.log1p(approved.clip(lower=0))*pd.to_numeric(x['exp98_pressure_12m'],errors='coerce');return x.drop(columns=['_month','_esc_change'],errors='ignore')
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_cost_oof(ctx['train'],ctx['cost_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];features=['production_prediction']+[f'exp98_{kind}_{w}m' for w in (3,6,12) for kind in ('pressure','dispersion')]+['exp98_size_exposure','approved_cost_cr','duration_ratio','cost_escalation_percentage'];corr,meta=fit_residual_booster(oof,score,features,9801);meta['external_wpi_used']=False;meta['proxy_definition']='lagged sector median/dispersion of observed cost-escalation changes';return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
