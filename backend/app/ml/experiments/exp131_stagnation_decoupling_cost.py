"""Exp131: leakage-safe stagnation/financial-decoupling cost residual challenger."""
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

def _metric(f,p): return float(_regression_metrics(f['actual_cost_overrun_percentage'],p,f['sample_weight'],f['canonical_project_id'])['MAE'])
def _gain(a,b): return (a-b)/a*100 if a else 0.0

def engineer(frame):
    x=frame.copy();x['snapshot_date']=pd.to_datetime(x['snapshot_date'],errors='coerce');x=x.sort_values(['canonical_project_id','snapshot_date'])
    g=x.groupby('canonical_project_id',sort=False)
    p=pd.to_numeric(x.get('physical_progress'),errors='coerce');e=pd.to_numeric(x.get('expenditure_ratio'),errors='coerce');sp=pd.to_numeric(x.get('cumulative_expenditure_cr'),errors='coerce');cost=pd.to_numeric(x.get('approved_cost_cr'),errors='coerce')
    pprev=g['physical_progress'].shift(1);d=(p-pd.to_numeric(pprev,errors='coerce')).fillna(0)
    inactive=(d<=0).astype(int);runs=[]
    for _,idx in x.groupby('canonical_project_id',sort=False).groups.items():
        c=0
        for i in idx:
            c=c+1 if inactive.loc[i] else 0;runs.append((i,c))
    r=pd.Series(dict(runs));x['stagnation_inactivity_snapshots']=r.reindex(x.index).fillna(0).astype(float)
    p6=pd.to_numeric(g['physical_progress'].shift(6),errors='coerce');e6=pd.to_numeric(g['expenditure_ratio'].shift(6),errors='coerce');s6=pd.to_numeric(g['cumulative_expenditure_cr'].shift(6),errors='coerce')
    dp=(p-p6)/100.0;de=e-e6
    x['trailing_decoupling_velocity_6m']=(de-dp).clip(lower=0).replace([np.inf,-np.inf],np.nan).fillna(0)
    denom=(cost*dp.clip(lower=.001)).clip(lower=.01);x['trailing_cost_burn_intensity_6m']=((sp-s6)/denom).replace([np.inf,-np.inf],np.nan).fillna(0)
    return x.sort_index()

def _design(train,score,features):
    A={};B={};cols=[]
    for c in features:
        if c not in train or c not in score: continue
        a=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan);b=pd.to_numeric(score[c],errors='coerce').replace([np.inf,-np.inf],np.nan);m=float(a.median()) if a.notna().any() else 0.0
        cols.append(c);A[c]=a.fillna(m);B[c]=b.fillna(m)
    return cols,pd.DataFrame(A,index=train.index),pd.DataFrame(B,index=score.index)

def _oof(data,identity,train,max_folds=4):
    cy=pd.to_numeric(train['completion_year'],errors='coerce');years=sorted(int(v) for v in cy.dropna().unique());parts=[]
    for y in years[-max_folds:]:
        val=train.loc[cy==y].copy();end=y-1
        if end<2005 or val['canonical_project_id'].nunique()<3: continue
        with tempfile.TemporaryDirectory(prefix=f'exp131-oof-{y}-') as td:
            root=Path(td)/'models';train_production(2001,end,y,data=data,identity=identity,artifact_root=root);m=joblib.load(root/f'2001_{end}'/'cost_model.pkl')
            val=engineer(val);val['production_prediction']=np.asarray(m.predict(val),float);val['residual']=pd.to_numeric(val['actual_cost_overrun_percentage'],errors='coerce').to_numpy(float)-val['production_prediction'].to_numpy(float);val['oof_year']=y;parts.append(val)
    if len(parts)<2: raise ValueError('Need >=2 forward OOF folds')
    return pd.concat(parts,ignore_index=True)

def fit_experiment(end=2021,output='reports/experiments/exp131_cost_2001_2021.json'):
    if end!=2021: raise ValueError('Exp131 audit is intentionally restricted to 2001-2021')
    data,identity=build_training_dataset();prepared=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(prepared,2001,end,2025);cohort=assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
    with tempfile.TemporaryDirectory(prefix='exp131-prod-') as td:
        root=Path(td)/'models';train_production(2001,end,2025,data=data,identity=identity,artifact_root=root);base=joblib.load(root/f'2001_{end}'/'cost_model.pkl');prod=np.asarray(base.predict(cohort),float)
    oof=_oof(data,identity,train);oof=engineer(oof);score=engineer(cohort)
    features=['production_prediction','stagnation_inactivity_snapshots','trailing_decoupling_velocity_6m','trailing_cost_burn_intensity_6m','cost_escalation_percentage','duration_ratio']
    years=sorted(oof.oof_year.unique());meta=[]
    for y in years[1:]:
        fit=oof[oof.oof_year<y];val=oof[oof.oof_year==y];cols,xf,xv=_design(fit,val,features);m=LGBMRegressor(n_estimators=120,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=50,reg_alpha=5,reg_lambda=20,random_state=131,verbosity=-1,n_jobs=1);m.fit(xf,fit.residual,sample_weight=fit.sample_weight);cap=max(float(np.nanquantile(np.abs(fit.residual),.9)),1e-9);meta.append((val,np.clip(m.predict(xv),-cap,cap)))
    best=(1e99,0.0)
    for s in (0,.25,.5,.75,1):
        vals=[]
        for v,c in meta: vals.append(np.average(np.abs(v.actual_cost_overrun_percentage-(v.production_prediction+s*c)),weights=v.sample_weight))
        best=min(best,(float(np.mean(vals)),float(s)))
    cols,xf,xs=_design(oof,score,features);m=LGBMRegressor(n_estimators=120,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=50,reg_alpha=5,reg_lambda=20,random_state=131,verbosity=-1,n_jobs=1);m.fit(xf,oof.residual,sample_weight=oof.sample_weight);cap=max(float(np.nanquantile(np.abs(oof.residual),.9)),1e-9);pred=prod+best[1]*np.clip(m.predict(xs),-cap,cap)
    pm=_metric(cohort,prod);em=_metric(cohort,pred);result={'experiment_id':'exp131','training_end':2021,'test_start':2022,'test_end':2025,'production_cost_mae':pm,'experiment_cost_mae':em,'cost_improvement_percentage':_gain(pm,em),'selected_scale':best[1],'features':cols,'holdout_used_for_selection':False,'scientific_verdict':'PROMOTION CANDIDATE' if em<pm else 'DO NOT PROMOTE'};p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(result),indent=2)+'\n');print(json.dumps(result,indent=2));return result
