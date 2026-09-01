"""Exp90: earned-value Cost-at-completion structural signals."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_90';EXPERIMENT_NAME='Earned-value Cost-at-completion ensemble';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=90

def _features(frame):
    x=frame.copy();sp=pd.to_numeric(x.get('expenditure_ratio'),errors='coerce');prog=pd.to_numeric(x.get('physical_progress'),errors='coerce')/100.0;time=pd.to_numeric(x.get('duration_ratio'),errors='coerce');approved=pd.to_numeric(x.get('approved_cost_cr'),errors='coerce');revised=pd.to_numeric(x.get('revised_cost_cr'),errors='coerce')
    cpi=prog/np.maximum(sp,0.02);spi=prog/np.maximum(time,0.05);x['exp90_cpi']=cpi.clip(0,5);x['exp90_spi']=spi.clip(0,5);x['exp90_eac_cpi_overrun']=((1/np.maximum(cpi,.1))-1)*100;x['exp90_eac_cpi_spi_overrun']=((1/np.maximum(cpi*spi,.1))-1)*100;x['exp90_revised_overrun']=np.where(approved>0,(revised/approved-1)*100,np.nan);x['exp90_spend_progress_gap']=sp-prog;x['exp90_progress_time_gap']=prog-time
    for c in ['exp90_eac_cpi_overrun','exp90_eac_cpi_spi_overrun','exp90_revised_overrun']: x[c]=pd.to_numeric(x[c],errors='coerce').clip(-100,400)
    return x

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=_features(current_cost_oof(ctx['train'],ctx['cost_model']));score=_features(ctx['cohort']);score['production_prediction']=ctx['production_cost'];features=['production_prediction','exp90_cpi','exp90_spi','exp90_eac_cpi_overrun','exp90_eac_cpi_spi_overrun','exp90_revised_overrun','exp90_spend_progress_gap','exp90_progress_time_gap','duration_ratio','schedule_slippage_days'];corr,meta=fit_residual_booster(oof,score,features,9001);meta['structural_method']='EVM-style CPI/SPI/EAC signals used only as bounded residual features';return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
