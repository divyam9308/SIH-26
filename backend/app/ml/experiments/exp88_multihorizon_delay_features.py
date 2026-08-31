"""Experiment 88: multi-horizon completion logits as U1 residual features."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from backend.app.ml.experiments.post_u1_delay_common import current_delay_oof,fit_residual_booster,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_88';EXPERIMENT_NAME='Multi-horizon completion logits as Delay residual features';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=88
HORIZONS=(180,365,730);BASE_FEATURES=['production_prediction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','elapsed_days','planned_duration_days','exp58_group_support']

def _remaining(frame):
    return (pd.to_datetime(frame['completion_date'],errors='coerce')-pd.to_datetime(frame['snapshot_date'],errors='coerce')).dt.days.clip(lower=0)

def _fit_probs(train,score):
    _,_,xt,xs=numeric_design(train,score,BASE_FEATURES);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);rem=_remaining(train);out={}
    for h in HORIZONS:
        y=(rem<=h).fillna(False).astype(int).to_numpy()
        if y.min()==y.max(): out[f'exp88_p_complete_{h}d']=np.full(len(score),float(y[0] if len(y) else 0));continue
        sc=StandardScaler();z=sc.fit_transform(xt);m=LogisticRegression(C=.1,max_iter=500,random_state=8800+h);m.fit(z,y,sample_weight=w);out[f'exp88_p_complete_{h}d']=m.predict_proba(sc.transform(xs))[:,1]
    return out

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);ys=pd.to_numeric(oof['oof_year'],errors='coerce');years=sorted(int(x) for x in ys.dropna().unique());parts=[]
    for year in years[1:]:
        fit=oof.loc[ys<year].copy();val=oof.loc[ys==year].copy()
        if len(fit)<100 or val.empty: continue
        p=val.copy();probs=_fit_probs(fit,val)
        for k,v in probs.items(): p[k]=v
        parts.append(p)
    if not parts: raise ValueError('No forward OOF multi-horizon logits')
    meta=pd.concat(parts,ignore_index=True);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));score['u1_correction']=ctx['production_delay']-base
    for k,v in _fit_probs(oof,score).items(): score[k]=v
    features=['production_prediction','u1_correction','exp88_p_complete_180d','exp88_p_complete_365d','exp88_p_complete_730d','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','exp58_group_support'];corr,details=fit_residual_booster(meta,score,features,8801);details['horizons_days']=list(HORIZONS);details['horizon_predictions_are_forward_oof_for_meta_training']=True;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_delay']+corr,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
