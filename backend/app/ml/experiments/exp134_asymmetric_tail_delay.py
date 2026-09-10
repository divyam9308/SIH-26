"""Exp134: asymmetric extreme-slippage tail specialist for Delay."""
from __future__ import annotations
import numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.post_exp113_delay_common import prepare_context,production_oof,numeric_design,persist

def caps(frame,base,mult=3.0):
 s=pd.to_numeric(frame['schedule_slippage_days'],errors='coerce').fillna(0).to_numpy(float);u=base*(1+mult*np.clip((s-365)/1825,0,1));return np.full(len(frame),-400.),u

def fit_experiment(end=2021,output='reports/experiments/exp134_delay_2001_2021.json'):
 if end!=2021: raise ValueError('Exp134 audit is restricted to 2001-2021')
 ctx=prepare_context(end);oof=production_oof(ctx,max_folds=6);score=ctx['cohort'].copy();score['production_prediction']=ctx['production_delay'];features=['production_prediction','duration_ratio','schedule_slippage_days','expenditure_ratio','cost_escalation_percentage','physical_progress'];yc=pd.to_numeric(oof.oof_year,errors='coerce');years=sorted(int(v) for v in yc.dropna().unique());meta=[]
 for y in years[1:]:
  fit=oof[yc<y];val=oof[yc==y];cols,_,xf,xv=numeric_design(fit,val,features);r=pd.to_numeric(fit.residual,errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(fit.sample_weight,errors='coerce').fillna(0).to_numpy(float);m=LGBMRegressor(objective='quantile',alpha=.65,n_estimators=120,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=50,reg_alpha=5,reg_lambda=25,random_state=134,verbosity=-1,n_jobs=1);m.fit(xf,r,sample_weight=w);base=max(float(np.nanquantile(np.abs(r),.9)),1e-9);lo,hi=caps(val,base);gate=(pd.to_numeric(val.schedule_slippage_days,errors='coerce').fillna(0).to_numpy(float)>365)&(pd.to_numeric(val.duration_ratio,errors='coerce').fillna(0).to_numpy(float)>1.15);c=np.where(gate,np.clip(m.predict(xv),lo,hi),0.);meta.append((val,c))
 best=(1e99,0.)
 for s in (0,.25,.5,.75,1):
  vals=[];ws=[]
  for v,c in meta:
   a=pd.to_numeric(v.actual_delay_days,errors='coerce').to_numpy(float);p=np.maximum(0,pd.to_numeric(v.production_prediction,errors='coerce').to_numpy(float)+s*c);w=pd.to_numeric(v.sample_weight,errors='coerce').to_numpy(float);vals.append(np.average(np.abs(a-p),weights=w));ws.append(w.sum())
  best=min(best,(float(np.average(vals,weights=ws)),float(s)))
 cols,_,xf,xs=numeric_design(oof,score,features);r=pd.to_numeric(oof.residual,errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(oof.sample_weight,errors='coerce').fillna(0).to_numpy(float);m=LGBMRegressor(objective='quantile',alpha=.65,n_estimators=120,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=50,reg_alpha=5,reg_lambda=25,random_state=134,verbosity=-1,n_jobs=1);m.fit(xf,r,sample_weight=w);base=max(float(np.nanquantile(np.abs(r),.9)),1e-9);lo,hi=caps(score,base);gate=(pd.to_numeric(score.schedule_slippage_days,errors='coerce').fillna(0).to_numpy(float)>365)&(pd.to_numeric(score.duration_ratio,errors='coerce').fillna(0).to_numpy(float)>1.15);corr=np.where(gate,np.clip(m.predict(xs),lo,hi),0.)*best[1];pred=np.maximum(0,ctx['production_delay']+corr);details={'selected_scale':best[1],'quantile_alpha':.65,'distress_gate':'slippage>365 and duration_ratio>1.15','base_cap':base,'upper_multiplier':3.0,'features':cols,'meta_oof_years':years[1:]};return persist('exp134','Asymmetric Extreme-Slippage Tail Specialist',ctx,pred,details,output)
