"""Exp96: fiscal-year execution-profile Cost residual features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_96';EXPERIMENT_NAME='Fiscal-year execution profile Cost calibration';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=96

def _add(frame):
    x=frame.copy();d=pd.to_datetime(x['snapshot_date'],errors='coerce');fm=((d.dt.month-4)%12)+1;x['exp96_fiscal_month']=fm;x['exp96_fiscal_q']=((fm-1)//3)+1;x['exp96_year_end']=fm.ge(10).astype(float);sp=pd.to_numeric(x.get('expenditure_ratio'),errors='coerce');dur=pd.to_numeric(x.get('duration_ratio'),errors='coerce');prog=pd.to_numeric(x.get('physical_progress'),errors='coerce')/100.0;x['exp96_spend_vs_fy']=sp-(fm/12.0);x['exp96_spend_vs_time']=sp-dur;x['exp96_progress_vs_fy']=prog-(fm/12.0);x['exp96_yearend_spend_pressure']=x['exp96_year_end']*sp;x['exp96_q_spend_interaction']=x['exp96_fiscal_q']*sp;return x
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=_add(current_cost_oof(ctx['train'],ctx['cost_model']));score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];score=_add(score);features=['production_prediction','exp96_fiscal_month','exp96_fiscal_q','exp96_year_end','exp96_spend_vs_fy','exp96_spend_vs_time','exp96_progress_vs_fy','exp96_yearend_spend_pressure','exp96_q_spend_interaction','cost_escalation_percentage','schedule_slippage_days'];corr,meta=fit_residual_booster(oof,score,features,9601);meta['fiscal_year_start_month']=4;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
