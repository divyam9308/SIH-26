"""Shared leakage-safe harness for post-Exp113 Delay experiments 120-129.

The fixed 688-project Exp35 audit gate is intentionally disabled only inside
this experiment harness. Production modules on main are not modified. For PR
comparisons, every project with valid as-of AFT evidence is eligible, and the
same experiment-only rule is applied to the freshly retrained production anchor
and challenger.
"""
from __future__ import annotations
import json,tempfile
from contextlib import contextmanager
from pathlib import Path
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.nextgen_common import _prepare,normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights,build_training_dataset
from backend.app.ml.monthly_training import _json_safe,_regression_metrics,temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
import backend.app.ml.production_exp35_baseline as _p35
import backend.app.ml.production_exp61_baseline as _p61
import backend.app.ml.production_u1_delay_baseline as _pu1
import backend.app.ml.production_exp105_exp113_baseline as _p105
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay as train_current_production
WINDOWS={2021:(2022,2025,721,11200),2022:(2023,2025,487,6870)}
def _experiment_aft_projects(frame,limit=None):
    required={'canonical_project_id','snapshot_date','planned_completion_date'}
    missing=sorted(required.difference(frame.columns))
    if missing: raise ValueError('Experiment AFT eligibility is missing required fields: '+', '.join(missing))
    work=frame[['canonical_project_id','snapshot_date','planned_completion_date']].copy();eligible=_p35.AFTResidualDelayModel._aft_eligible(work)
    return set(work.loc[eligible,'canonical_project_id'].astype('string').tolist())
@contextmanager
def _experiment_aft_compat():
    modules=(_p35,_p61,_pu1,_p105);previous=[getattr(m,'_select_aft_calibration_projects',None) for m in modules]
    for m in modules: m._select_aft_calibration_projects=_experiment_aft_projects
    try: yield
    finally:
        for m,old in zip(modules,previous):
            if old is None: delattr(m,'_select_aft_calibration_projects')
            else: m._select_aft_calibration_projects=old
def window_contract(end):
    if end not in WINDOWS: raise ValueError('Only 2001-2021 and 2001-2022 are allowed')
    return WINDOWS[end]
def metric(f,a,p): return float(_regression_metrics(f[a],p,f['sample_weight'],f['canonical_project_id'])['MAE'])
def gain(a,b): return (float(a)-float(b))/float(a)*100 if float(a) else 0.0
def numeric_design(train,score,features):
    cols=[c for c in features if c in train.columns and c in score.columns];A={};B={};med={}
    for c in cols:
        x=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan);y=pd.to_numeric(score[c],errors='coerce').replace([np.inf,-np.inf],np.nan);m=float(x.median()) if x.notna().any() else 0.0;med[c]=m;A[c]=x.fillna(m);B[c]=y.fillna(m)
    return cols,med,pd.DataFrame(A,index=train.index),pd.DataFrame(B,index=score.index)
def forward_folds(frame,max_folds=8):
    cy=pd.to_numeric(frame['completion_year'],errors='coerce');years=sorted(int(x) for x in cy.dropna().unique());out=[]
    for y in reversed(years[1:]):
        fit=frame.loc[cy<y].copy();val=frame.loc[cy==y].copy()
        if fit['canonical_project_id'].nunique()>=10 and val['canonical_project_id'].nunique()>=3: out.append((fit,val,y))
        if len(out)>=max_folds: break
    return list(reversed(out))
def prepare_context(end):
    test_start,test_end,projects,snapshots=window_contract(end);data,identity=build_training_dataset()
    with tempfile.TemporaryDirectory(prefix=f'post-exp113-{end}-') as td:
        root=Path(td)/'models'
        with _experiment_aft_compat(): train_current_production(2001,end,test_end,data=data,identity=identity,artifact_root=root)
        target=root/f'2001_{end}';cm=joblib.load(target/'cost_model.pkl');dm=joblib.load(target/'delay_model.pkl');prepared=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(prepared,2001,end,test_end);train,test,_=_build_temporal_delay_priors(train,test)
        cohort=_production_cost_evaluation_rows(test).copy();ids=_experiment_aft_projects(cohort);cohort[CALIBRATION_GATE_FEATURE]=cohort['canonical_project_id'].astype('string').isin(ids);cohort=assign_project_balanced_weights(cohort)
        if int(cohort['canonical_project_id'].nunique())!=projects or len(cohort)!=snapshots: raise RuntimeError('Comparison cohort changed')
        pc=np.asarray(cm.predict(cohort),float);pdly=np.maximum(0,np.asarray(dm.predict(cohort),float))
    return dict(training_end=end,test_start=test_start,test_end=test_end,train=train,cohort=cohort,cost_model=cm,delay_model=dm,production_cost=pc,production_delay=pdly,full_data=data,identity=identity,aft_compatibility='all projects with valid as-of AFT evidence; fixed 688 audit gate disabled in PR only')
def production_oof(ctx,max_folds=6):
    parts=[];data=ctx['train']
    for fit,val,year in forward_folds(data,max_folds):
        train_end=int(year)-1
        if train_end<2005: continue
        try:
            with tempfile.TemporaryDirectory(prefix=f'prod-oof-{year}-') as td:
                root=Path(td)/'models'
                with _experiment_aft_compat(): train_current_production(2001,train_end,int(year),data=ctx['full_data'],identity=ctx['identity'],artifact_root=root)
                dm=joblib.load(root/f'2001_{train_end}'/'delay_model.pkl');p=np.maximum(0,np.asarray(dm.predict(val),float))
        except Exception: continue
        part=val.copy();part['production_prediction']=p;part['residual']=pd.to_numeric(part['actual_delay_days'],errors='coerce').to_numpy(float)-p;part['oof_year']=int(year);parts.append(part)
    if len(parts)<2: raise ValueError('Need >=2 strict forward production OOF folds')
    return pd.concat(parts,ignore_index=True)
def fit_residual(oof,score,features,seed,extra_weight=None,monotone=None):
    yc=pd.to_numeric(oof['oof_year'],errors='coerce');years=sorted(int(x) for x in yc.dropna().unique());meta=[]
    for y in years[1:]:
        fit=oof.loc[yc<y].copy();val=oof.loc[yc==y].copy()
        if len(fit)<80 or val.empty: continue
        cols,_,xf,xv=numeric_design(fit,val,features);kwargs={}
        if monotone: kwargs['monotone_constraints']=[int(monotone.get(c,0)) for c in cols]
        m=LGBMRegressor(n_estimators=100,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=60,reg_alpha=5,reg_lambda=25,random_state=seed,verbosity=-1,n_jobs=1,**kwargs);r=pd.to_numeric(fit['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(fit['sample_weight'],errors='coerce').fillna(0).to_numpy(float)
        if extra_weight is not None: w=w*np.asarray(extra_weight(fit),float)
        m.fit(xf,r,sample_weight=w);cap=max(float(np.nanquantile(np.abs(r),.9)),1e-9);meta.append((val,np.clip(np.asarray(m.predict(xv),float),-cap,cap)))
    if not meta: raise ValueError('No forward residual predictions')
    best=(float('inf'),0.0)
    for s in (0,.25,.5,.75,1.0):
        vals=[];weights=[]
        for val,c in meta:
            a=pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float);p=np.maximum(0,pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float)+s*c);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);vals.append(float(np.average(np.abs(a-p),weights=w)));weights.append(max(float(w.sum()),1e-9))
        best=min(best,(float(np.average(vals,weights=weights)),float(s)))
    scale=best[1];cols,med,xf,xs=numeric_design(oof,score,features);kwargs={}
    if monotone: kwargs['monotone_constraints']=[int(monotone.get(c,0)) for c in cols]
    m=LGBMRegressor(n_estimators=160,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=60,reg_alpha=5,reg_lambda=25,random_state=seed,verbosity=-1,n_jobs=1,**kwargs);r=pd.to_numeric(oof['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(oof['sample_weight'],errors='coerce').fillna(0).to_numpy(float)
    if extra_weight is not None: w=w*np.asarray(extra_weight(oof),float)
    m.fit(xf,r,sample_weight=w);cap=max(float(np.nanquantile(np.abs(r),.9)),1e-9);corr=scale*np.clip(np.asarray(m.predict(xs),float),-cap,cap);return corr,dict(selected_scale=scale,features=cols,cap=cap,meta_oof_years=years[1:])
def persist(exp_id,name,ctx,pred,details,output):
    c=ctx['cohort'];pc=ctx['production_cost'];pdly=ctx['production_delay'];ed=np.maximum(0,np.asarray(pred,float));pcm=metric(c,'actual_cost_overrun_percentage',pc);pdm=metric(c,'actual_delay_days',pdly);edm=metric(c,'actual_delay_days',ed);dg=gain(pdm,edm);verdict='PROMOTION CANDIDATE' if dg>0 else 'DO NOT PROMOTE';details=dict(details);details['aft_pr_compatibility']=ctx['aft_compatibility'];result={'experiment_id':exp_id,'experiment_name':name,'scope':'delay','training_start':2001,'training_end':ctx['training_end'],'test_start':ctx['test_start'],'test_end':ctx['test_end'],'production_cost_mae':pcm,'experiment_cost_mae':pcm,'cost_improvement_percentage':0.0,'production_delay_mae':pdm,'experiment_delay_mae':edm,'delay_improvement_percentage':round(dg,6),'comparison_test_projects':int(c['canonical_project_id'].nunique()),'comparison_test_snapshots':len(c),'cost_predictions_identical':True,'holdout_used_for_selection':False,'promotion_allowed':False,'execution_verdict':'EXECUTION VALID','scientific_verdict':verdict,'details':_json_safe(details)};p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(result),indent=2,allow_nan=False)+'\n');print(f'{exp_id.upper()}_2001_{ctx["training_end"]}_PRODUCTION_DELAY_MAE={pdm:.6f}');print(f'{exp_id.upper()}_2001_{ctx["training_end"]}_EXPERIMENT_DELAY_MAE={edm:.6f}');print(f'{exp_id.upper()}_2001_{ctx["training_end"]}_DELAY_IMPROVEMENT_PERCENT={dg:.6f}');print(f'{exp_id.upper()}_SCIENTIFIC_VERDICT={verdict}');return result
