"""Experiment 80: revision-shock Cost residual features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_80';EXPERIMENT_NAME='Revision-shock residual Cost calibration';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=80

def engineer(frame):
    out=frame.copy();out['_exp80_order']=np.arange(len(out));out=out.sort_values(['canonical_project_id','snapshot_date'])
    rev=pd.to_numeric(out.get('revised_cost_cr'),errors='coerce');approved=pd.to_numeric(out.get('approved_cost_cr'),errors='coerce').replace(0,np.nan);g=out['canonical_project_id']
    delta=rev.groupby(g).diff();shock=100.0*delta/approved;event=delta.abs().fillna(0).gt(1e-9)
    out['exp80_revision_shock_pct']=shock.replace([np.inf,-np.inf],np.nan);out['exp80_revision_event']=event.astype(float);out['exp80_revision_count']=event.groupby(g).cumsum().astype(float);out['exp80_max_abs_shock_pct']=out['exp80_revision_shock_pct'].abs().groupby(g).cummax()
    dates=pd.to_datetime(out['snapshot_date'],errors='coerce');last=dates.where(event).groupby(g).ffill();out['exp80_days_since_revision']=(dates-last).dt.days.astype(float)
    return out.sort_values('_exp80_order').drop(columns=['_exp80_order'])

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer);oof=current_cost_oof(ctx['train'],ctx['cost_model']);features=['production_prediction','exp80_revision_shock_pct','exp80_revision_event','exp80_revision_count','exp80_max_abs_shock_pct','exp80_days_since_revision','cost_escalation_percentage','expenditure_ratio','duration_ratio','schedule_slippage_days']
    corr,meta=fit_residual_booster(oof,ctx['cohort'],features,8001);pred=ctx['production_cost']+corr;meta['prefix_only_revision_features']=True;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,meta,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
