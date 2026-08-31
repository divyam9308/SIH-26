"""Experiment 85: U1 plus schedule-revision shock residual features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_85';EXPERIMENT_NAME='U1 plus schedule-revision shock residual booster';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=85

def engineer(frame):
    out=frame.copy();out['_ord']=np.arange(len(out));out=out.sort_values(['canonical_project_id','snapshot_date']);g=out['canonical_project_id'];rev=pd.to_datetime(out.get('revised_completion_date'),errors='coerce');delta=rev.groupby(g).diff().dt.days.astype(float);event=delta.abs().fillna(0).gt(0);out['exp85_schedule_revision_days']=delta;out['exp85_abs_schedule_revision_days']=delta.abs();out['exp85_schedule_revision_count']=event.groupby(g).cumsum().astype(float);out['exp85_max_extension_days']=delta.clip(lower=0).groupby(g).cummax();dates=pd.to_datetime(out['snapshot_date'],errors='coerce');last=dates.where(event).groupby(g).ffill();out['exp85_days_since_schedule_revision']=(dates-last).dt.days.astype(float);return out.sort_values('_ord').drop(columns=['_ord'])

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end,engineer);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];score['u1_correction']=ctx['production_delay']-np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));features=['production_prediction','u1_correction','exp85_schedule_revision_days','exp85_abs_schedule_revision_days','exp85_schedule_revision_count','exp85_max_extension_days','exp85_days_since_schedule_revision','schedule_slippage_days','duration_ratio','expenditure_ratio','exp58_delay_hier_prior','exp58_group_support'];corr,meta=fit_residual_booster(oof,score,features,8501);meta['prefix_only_schedule_revision_features']=True;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,meta,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
