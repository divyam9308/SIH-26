"""Exp107: lagged relational cross-project Cost features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_107';EXPERIMENT_NAME='Cross-project relational Cost features';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=107
SIGNALS=('cost_escalation_percentage','schedule_slippage_days','expenditure_ratio')
GROUPS=('implementing_agency','sector','state')
def _engineer(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x['_month']=x['snapshot_date'].dt.to_period('M').dt.to_timestamp()
    for key in GROUPS:
        if key not in x.columns:continue
        agg=x.groupby([key,'_month'],dropna=False).agg(**{f'_m_{s}':(s,'median') for s in SIGNALS},_n=('canonical_project_id','nunique')).reset_index().sort_values([key,'_month'])
        made=[]
        for sig in SIGNALS:
            src=f'_m_{sig}';col=f'exp107_{key}_{sig}_6m';agg[col]=agg.groupby(key)[src].transform(lambda s:s.shift(1).rolling(6,min_periods=1).median());made.append(col)
        ncol=f'exp107_{key}_support_6m';agg[ncol]=agg.groupby(key)['_n'].transform(lambda s:s.shift(1).rolling(6,min_periods=1).mean());made.append(ncol);x=x.merge(agg[[key,'_month']+made],on=[key,'_month'],how='left',sort=False)
    return x.drop(columns=['_month'],errors='ignore')
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_cost_oof(ctx['train'],ctx['cost_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];features=['production_prediction']+[f'exp107_{g}_{s}_6m' for g in GROUPS for s in SIGNALS]+[f'exp107_{g}_support_6m' for g in GROUPS]+['duration_ratio','cost_escalation_percentage','approved_cost_cr'];corr,meta=fit_residual_booster(oof,score,features,10701);meta['relation_types']=list(GROUPS);meta['aggregates_lagged_one_month']=True;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
