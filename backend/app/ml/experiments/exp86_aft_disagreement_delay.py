"""Experiment 86: AFT family disagreement as a residual signal beyond U1."""
from __future__ import annotations
import argparse,numpy as np
from backend.app.ml.experiments.post_u1_delay_common import aft_disagreement,current_delay_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_86';EXPERIMENT_NAME='AFT disagreement residual signal';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=86

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base;score['aft_disagreement']=aft_disagreement(ctx['delay_model'],score)
    features=['production_prediction','u1_correction','aft_disagreement','schedule_slippage_days','duration_ratio','expenditure_ratio','cost_escalation_percentage','exp58_delay_hier_prior','exp58_group_support'];corr,meta=fit_residual_booster(oof,score,features,8601);meta['aft_disagreement_definition']='std of LightGBM/XGBoost/ExtraTrees raw AFT delay predictions';return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,meta,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
