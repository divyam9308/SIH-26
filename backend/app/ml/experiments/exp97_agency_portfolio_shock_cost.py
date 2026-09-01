"""Exp97: lagged agency portfolio Cost-shock index."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_97';EXPERIMENT_NAME='Agency portfolio Cost-shock index';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=97

def _engineer(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x['_month']=x['snapshot_date'].dt.to_period('M').dt.to_timestamp();x=x.sort_values(['canonical_project_id','snapshot_date']);rev=pd.to_numeric(x['revised_cost_cr'],errors='coerce').fillna(pd.to_numeric(x['approved_cost_cr'],errors='coerce'));x['_rev_jump']=x.groupby('canonical_project_id',sort=False)[rev.name].pct_change() if rev.name in x.columns else np.nan
    # pct_change above cannot use the filled series directly; compute explicitly from aligned series.
    prev=rev.groupby(x['canonical_project_id']).shift(1);x['_rev_jump']=np.where(prev.abs()>1e-9,(rev-prev)/prev,np.nan);x['_upshock']=(pd.Series(x['_rev_jump'],index=x.index)>0.02).astype(float)
    monthly=x.groupby(['implementing_agency','_month'],dropna=False).agg(_share=('_upshock','mean'),_jump=('_rev_jump','median'),_n=('canonical_project_id','nunique')).reset_index().sort_values(['implementing_agency','_month'])
    for win in (3,6,12):
        monthly[f'exp97_share_up_{win}m']=monthly.groupby('implementing_agency')['_share'].transform(lambda s:s.shift(1).rolling(win,min_periods=1).mean());monthly[f'exp97_jump_{win}m']=monthly.groupby('implementing_agency')['_jump'].transform(lambda s:s.shift(1).rolling(win,min_periods=1).median());monthly[f'exp97_projects_{win}m']=monthly.groupby('implementing_agency')['_n'].transform(lambda s:s.shift(1).rolling(win,min_periods=1).mean())
    keep=['implementing_agency','_month']+[c for c in monthly.columns if c.startswith('exp97_')];x=x.merge(monthly[keep],on=['implementing_agency','_month'],how='left',sort=False);return x.drop(columns=['_month','_rev_jump','_upshock'],errors='ignore')
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_cost_oof(ctx['train'],ctx['cost_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];features=['production_prediction']+[f'exp97_{kind}_{w}m' for w in (3,6,12) for kind in ('share_up','jump','projects')]+['cost_escalation_percentage','duration_ratio','schedule_slippage_days'];corr,meta=fit_residual_booster(oof,score,features,9701);meta['portfolio_features_lagged_one_month']=True;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
