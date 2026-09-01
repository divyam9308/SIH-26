"""Exp104: causal denoising of noisy/stale project measurement state."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_104';EXPERIMENT_NAME='Latent measurement-state Cost denoising';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=104
BASE=('cost_escalation_percentage','expenditure_ratio','physical_progress','schedule_slippage_days','progress_deviation')
def _engineer(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x=x.sort_values(['canonical_project_id','snapshot_date']).copy();g=x.groupby('canonical_project_id',sort=False)
    x['exp104_report_gap_days']=g['snapshot_date'].diff().dt.days
    for c in BASE:
        s=pd.to_numeric(x[c],errors='coerce');x[f'exp104_{c}_level']=s.groupby(x['canonical_project_id']).transform(lambda z:z.ewm(alpha=.35,adjust=False,min_periods=1).mean());x[f'exp104_{c}_innovation']=s-x[f'exp104_{c}_level'];x[f'exp104_{c}_volatility']=x[f'exp104_{c}_innovation'].groupby(x['canonical_project_id']).transform(lambda z:z.ewm(alpha=.25,adjust=False,min_periods=2).std())
    return x.sort_index()
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer=_engineer);oof=current_cost_oof(ctx['train'],ctx['cost_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];features=['production_prediction','exp104_report_gap_days']+[f'exp104_{c}_{k}' for c in BASE for k in ('level','innovation','volatility')]+['duration_ratio','approved_cost_cr'];corr,meta=fit_residual_booster(oof,score,features,10401);meta.update({'filter':'causal exponential latent-state proxy','level_alpha':.35,'volatility_alpha':.25});return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
