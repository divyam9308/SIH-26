"""Exp132: scale-conditioned cost residual challenger; audit restricted to 2001-2021."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.nextgen_common import _prepare,normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights,build_training_dataset
from backend.app.ml.monthly_training import _json_safe,_regression_metrics,temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay as train_production

def metric(f,p): return float(_regression_metrics(f['actual_cost_overrun_percentage'],p,f['sample_weight'],f['canonical_project_id'])['MAE'])
def add_scale(f,priors=None):
 x=f.copy();c=pd.to_numeric(x['approved_cost_cr'],errors='coerce').clip(lower=0);d=pd.to_numeric(x['planned_duration_days'],errors='coerce').clip(lower=30);x['log_approved_cost']=np.log1p(c);x['capital_intensity_cr_per_day']=c/d
 if priors is None:
  med=x.assign(_c=c).groupby('sector')['_c'].median().to_dict();glob=float(c.median())
 else: med,glob=priors
 den=x['sector'].map(med).fillna(glob).replace(0,glob if glob else 1);x['sector_relative_cost_ratio']=c/den
 return x,(med,glob)
def design(a,b,fs):
 A={};B={};cols=[]
 for c in fs:
  if c not in a or c not in b: continue
  x=pd.to_numeric(a[c],errors='coerce').replace([np.inf,-np.inf],np.nan);y=pd.to_numeric(b[c],errors='coerce').replace([np.inf,-np.inf],np.nan);m=float(x.median()) if x.notna().any() else 0.;cols.append(c);A[c]=x.fillna(m);B[c]=y.fillna(m)
 return cols,pd.DataFrame(A,index=a.index),pd.DataFrame(B,index=b.index)
def fit_experiment(end=2021,output='reports/experiments/exp132_cost_2001_2021.json'):
 if end!=2021: raise ValueError('Exp132 audit is restricted to 2001-2021')
 data,identity=build_training_dataset();p=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(p,2001,2021,2025);cohort=assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
 with tempfile.TemporaryDirectory(prefix='exp132-prod-') as td:
  root=Path(td)/'models';train_production(2001,2021,2025,data=data,identity=identity,artifact_root=root);m=joblib.load(root/'2001_2021'/'cost_model.pkl');prod=np.asarray(m.predict(cohort),float)
 cy=pd.to_numeric(train['completion_year'],errors='coerce');years=sorted(int(v) for v in cy.dropna().unique())[-4:];parts=[]
 for y in years:
  val=train[cy==y].copy();te=y-1
  if te<2005 or val['canonical_project_id'].nunique()<3: continue
  with tempfile.TemporaryDirectory(prefix=f'exp132-{y}-') as td:
   root=Path(td)/'models';train_production(2001,te,y,data=data,identity=identity,artifact_root=root);bm=joblib.load(root/f'2001_{te}'/'cost_model.pkl');val['production_prediction']=bm.predict(val);val['residual']=val.actual_cost_overrun_percentage-val.production_prediction;val['oof_year']=y;parts.append(val)
 oof=pd.concat(parts,ignore_index=True);oof,pri=add_scale(oof);score,_=add_scale(cohort,pri);fs=['production_prediction','cost_escalation_percentage','duration_ratio','log_approved_cost','capital_intensity_cr_per_day','sector_relative_cost_ratio'];yrs=sorted(oof.oof_year.unique());meta=[]
 for y in yrs[1:]:
  fit=oof[oof.oof_year<y];val=oof[oof.oof_year==y];cols,xf,xv=design(fit,val,fs);lm=LGBMRegressor(objective='huber',alpha=.9,n_estimators=100,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=50,random_state=132,verbosity=-1,n_jobs=1);lm.fit(xf,fit.residual,sample_weight=fit.sample_weight);base=max(float(np.nanquantile(np.abs(fit.residual),.9)),1e-9);ratio=np.clip(pd.to_numeric(val.log_approved_cost)/max(float(pd.to_numeric(fit.log_approved_cost).median()),1e-9),.5,2.0);meta.append((val,np.clip(lm.predict(xv),-base*(.8+.4*ratio),base*(.8+.4*ratio))))
 best=(1e99,0.)
 for s in (0,.25,.5,.75,1): best=min(best,(float(np.mean([np.average(np.abs(v.actual_cost_overrun_percentage-(v.production_prediction+s*c)),weights=v.sample_weight) for v,c in meta])),float(s)))
 cols,xf,xs=design(oof,score,fs);lm=LGBMRegressor(objective='huber',alpha=.9,n_estimators=100,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=50,random_state=132,verbosity=-1,n_jobs=1);lm.fit(xf,oof.residual,sample_weight=oof.sample_weight);base=max(float(np.nanquantile(np.abs(oof.residual),.9)),1e-9);ratio=np.clip(pd.to_numeric(score.log_approved_cost)/max(float(pd.to_numeric(oof.log_approved_cost).median()),1e-9),.5,2.0);corr=np.clip(lm.predict(xs),-base*(.8+.4*ratio),base*(.8+.4*ratio));pred=prod+best[1]*corr;pm=metric(cohort,prod);em=metric(cohort,pred);r={'experiment_id':'exp132','training_end':2021,'production_cost_mae':pm,'experiment_cost_mae':em,'cost_improvement_percentage':(pm-em)/pm*100,'selected_scale':best[1],'holdout_used_for_selection':False,'scientific_verdict':'PROMOTION CANDIDATE' if em<pm else 'DO NOT PROMOTE'};Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(_json_safe(r),indent=2)+'\n');print(json.dumps(r,indent=2));return r
