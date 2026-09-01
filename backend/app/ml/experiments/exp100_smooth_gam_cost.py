"""Exp100: smooth additive OOF Cost residual calibration."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import QuantileRegressor
from backend.app.ml.experiments.post_u1_cost_common import _mae,_wq,current_cost_oof,numeric_design,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_100';EXPERIMENT_NAME='Smooth OOF GAM Cost residual calibration';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=100
FEATURES=['production_prediction','cost_escalation_percentage','expenditure_ratio','duration_ratio','progress_deviation','schedule_slippage_days','approved_cost_cr']

def _fit(train,score):
    _,_,xt,xs=numeric_design(train,score,FEATURES);sp=SplineTransformer(n_knots=4,degree=2,include_bias=False);zt=sp.fit_transform(xt);zs=sp.transform(xs);r=pd.to_numeric(train['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);m=QuantileRegressor(quantile=.5,alpha=.08,solver='highs');m.fit(zt,r,sample_weight=w);cap=max(_wq(np.abs(r),w,.9),1e-9);return np.clip(np.asarray(m.predict(zs),float),-cap,cap),cap

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);yy=pd.to_numeric(oof['oof_year'],errors='coerce');meta=[]
    for year in sorted(int(v) for v in yy.dropna().unique())[1:]:
        fit=oof.loc[yy<year];val=oof.loc[yy==year]
        if len(fit)<80 or val.empty: continue
        c,_=_fit(fit,val);meta.append((val,c))
    if not meta: raise ValueError('No spline meta-OOF folds')
    best=(float('inf'),0.0)
    for scale in (0.0,.25,.5,.75,1.0):
        vals=[];ws=[]
        for val,c in meta:
            p=val['production_prediction'].to_numpy(float)+scale*c;vals.append(_mae(val['actual_cost_overrun_percentage'],p,val['sample_weight']));ws.append(float(val['sample_weight'].sum()))
        best=min(best,(float(np.average(vals,weights=ws)),scale))
    score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];corr,cap=_fit(oof,score);scale=best[1];details={'features':FEATURES,'basis':'quadratic splines','knots':4,'quantile':.5,'alpha':.08,'selected_scale':scale,'cap':cap};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+scale*corr,details,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
