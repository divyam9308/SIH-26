"""Shared isolated Delay harness for post-PR110 experiments.

Freshly retrains current production (Exp61 + U1 Delay), constructs deeper
forward OOF predictions of that same Delay stack, and evaluates Delay-only
challengers on the exact standard 2019/2021 future cohorts.
"""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.exp35_aft_residual_combo import _aft_remaining_prediction,_corrections,_delay_from_remaining,_fit_aft_family_models,_remaining_frame
from backend.app.ml.experiments.nextgen_common import _hash_prod,_prepare,normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights,build_training_dataset
from backend.app.ml.monthly_training import _json_safe,_regression_metrics,temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel,CALIBRATION_GATE_FEATURE,_select_aft_calibration_projects
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_u1_delay_baseline import _fit_u1_booster,train_window_with_promoted_cost_and_delay as train_current_production
WINDOWS={2019:(2020,2025,949,14847),2021:(2022,2025,721,11200)}

def window_contract(end):
    if end not in WINDOWS: raise ValueError('Only 2001-2019 and 2001-2021 are allowed')
    return WINDOWS[end]
def _metric(f,a,p): return float(_regression_metrics(f[a],p,f['sample_weight'],f['canonical_project_id'])['MAE'])
def _gain(a,b): return (float(a)-float(b))/float(a)*100 if float(a) else 0.0

def _wq(v,w,q):
    v=np.asarray(v,float);w=np.asarray(w,float);m=np.isfinite(v)&np.isfinite(w)&(w>=0);v=v[m];w=w[m]
    if not len(v): return 0.0
    o=np.argsort(v);v=v[o];w=w[o];s=float(w.sum());return float(np.quantile(v,q)) if s<=0 else float(v[np.searchsorted(np.cumsum(w),q*s,side='left')])
def _mae(y,p,w):
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float);m=np.isfinite(y)&np.isfinite(p)&np.isfinite(w)&(w>=0);return float(np.average(np.abs(y[m]-p[m]),weights=w[m])) if float(w[m].sum())>0 else float(np.mean(np.abs(y[m]-p[m])))

def forward_folds(frame,max_folds=8):
    cy=pd.to_numeric(frame['completion_year'],errors='coerce');years=sorted(int(x) for x in cy.dropna().unique());out=[]
    for year in reversed(years[1:]):
        fit=frame.loc[cy<year].copy();val=frame.loc[cy==year].copy()
        if fit['canonical_project_id'].nunique()>=10 and val['canonical_project_id'].nunique()>=3: out.append((fit,val,year))
        if len(out)>=max_folds: break
    return list(reversed(out))

def _family_disagreement(models,weights,frame,features):
    """Std-dev of family Delay predictions without invoking the full-family combiner.

    ``_aft_remaining_prediction`` always iterates the repository-wide FAMILIES
    constant, so passing it a one-model dict raises KeyError for the other
    families. For disagreement we intentionally score each fitted family alone:
    model.predict returns log1p(remaining_days), which is converted back exactly
    before the normal remaining->Delay conversion.
    """
    if not len(frame): return np.zeros(0,float)
    arr=[]
    for _name,model in models.items():
        log_remaining=np.asarray(model.predict(frame[features]),float)
        remaining=np.maximum(0.0,np.expm1(np.clip(log_remaining,-20,20)))
        arr.append(_delay_from_remaining(frame,remaining))
    if not arr: return np.zeros(len(frame),float)
    stacked=np.vstack(arr)
    if stacked.shape[1]!=len(frame) or not np.isfinite(stacked).all():
        raise AssertionError('AFT family disagreement produced invalid predictions')
    return np.std(stacked,axis=0)

def _base_oof(train,u1_model):
    if not hasattr(u1_model,'base_model'): raise TypeError('Post-PR110 U1 Delay wrapper required')
    base=u1_model.base_model;features=list(base.model_features);td=_remaining_frame(train);out=[]
    for fit,val,year in forward_folds(td,8):
        models=_fit_aft_family_models(fit,features);rem=_aft_remaining_prediction(models,base.weights,val,features);raw=_delay_from_remaining(val,rem);pred=np.maximum(0,raw+_corrections(val,raw,base.calibration));part=val.copy();part['base_prediction']=pred;part['production_prediction']=pred;part['aft_disagreement']=_family_disagreement(models,base.weights,val,features);part['residual']=pd.to_numeric(part['actual_delay_days'],errors='coerce').to_numpy(float)-pred;part['oof_year']=year;out.append(part)
    if len(out)<4: raise ValueError('Need >=4 base Delay folds')
    return pd.concat(out,ignore_index=True)

def current_delay_oof(train,u1_model):
    base=_base_oof(train,u1_model);ys=pd.to_numeric(base['oof_year'],errors='coerce');years=sorted(int(x) for x in ys.dropna().unique());out=[]
    for year in years[1:]:
        fit=base.loc[ys<year].copy();val=base.loc[ys==year].copy()
        if len(fit)<100 or val.empty: continue
        _,_,_,_,corr=_fit_u1_booster(fit,val);anchor=pd.to_numeric(val['base_prediction'],errors='coerce').to_numpy(float);pred=np.maximum(0,anchor+corr);part=val.copy();part['production_prediction']=pred;part['u1_correction']=pred-anchor;part['residual']=pd.to_numeric(part['actual_delay_days'],errors='coerce').to_numpy(float)-pred;part['oof_year']=year;out.append(part)
    if len(out)<3: raise ValueError('Need >=3 current-production Delay OOF folds')
    return pd.concat(out,ignore_index=True)

def aft_disagreement(delay_model,frame):
    base=delay_model.base_model;work=base._enrich(frame.copy()) if hasattr(base,'_enrich') else frame.copy();eligible=AFTResidualDelayModel._aft_eligible(work).to_numpy(bool);out=np.zeros(len(work),float);pos=np.flatnonzero(eligible)
    if len(pos): out[pos]=_family_disagreement(base.aft_models,base.weights,work.iloc[pos],list(base.model_features))
    return out

def numeric_design(train,score,features):
    cols=[c for c in features if c in train.columns and c in score.columns];a={};b={};med={}
    for c in cols:
        x=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan);y=pd.to_numeric(score[c],errors='coerce').replace([np.inf,-np.inf],np.nan);m=float(x.median()) if x.notna().any() else 0.0;a[c]=x.fillna(m);b[c]=y.fillna(m);med[c]=m
    return cols,med,pd.DataFrame(a,index=train.index),pd.DataFrame(b,index=score.index)

def fit_residual_booster(oof,score,features,seed):
    years=sorted(int(x) for x in pd.to_numeric(oof['oof_year'],errors='coerce').dropna().unique());meta=[]
    for year in years[1:]:
        yy=pd.to_numeric(oof['oof_year'],errors='coerce');fit=oof.loc[yy<year].copy();val=oof.loc[yy==year].copy()
        if len(fit)<100 or val.empty: continue
        _,_,xf,xv=numeric_design(fit,val,features);m=LGBMRegressor(n_estimators=120,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=60,reg_alpha=4,reg_lambda=20,random_state=seed,verbosity=-1,n_jobs=1);r=pd.to_numeric(fit['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(fit['sample_weight'],errors='coerce').fillna(0).to_numpy(float);m.fit(xf,r,sample_weight=w);cap=max(_wq(np.abs(r),w,.9),1e-9);meta.append((val,np.clip(np.asarray(m.predict(xv),float),-cap,cap)))
    if not meta: raise ValueError('No Delay meta-OOF residual predictions')
    best=(float('inf'),0.0)
    for scale in (0.0,.25,.5,.75,1.0):
        vals=[];weights=[]
        for val,c in meta:
            p=np.maximum(0,pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float)+scale*c);y=pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);vals.append(_mae(y,p,w));weights.append(max(float(w.sum()),1e-9))
        if vals: best=min(best,(float(np.average(vals,weights=weights)),scale))
    scale=best[1];cols,med,xf,xs=numeric_design(oof,score,features);m=LGBMRegressor(n_estimators=160,learning_rate=.025,max_depth=3,num_leaves=8,min_child_samples=60,reg_alpha=4,reg_lambda=20,random_state=seed,verbosity=-1,n_jobs=1);r=pd.to_numeric(oof['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(oof['sample_weight'],errors='coerce').fillna(0).to_numpy(float);m.fit(xf,r,sample_weight=w);cap=max(_wq(np.abs(r),w,.9),1e-9);corr=scale*np.clip(np.asarray(m.predict(xs),float),-cap,cap);return corr,{'selected_scale':scale,'features':cols,'medians':med,'cap':cap,'meta_oof_years':years[1:]}

def prepare_context(end,engineer=None):
    test_start,test_end,projects,snapshots=window_contract(end);data,identity=build_training_dataset();before=_hash_prod()
    with tempfile.TemporaryDirectory(prefix=f'exp-delay-{end}-') as td:
        root=Path(td)/'models';train_current_production(2001,end,test_end,data=data,identity=identity,artifact_root=root);target=root/f'2001_{end}';cm=joblib.load(target/'cost_model.pkl');dm=joblib.load(target/'delay_model.pkl');prepared=normalize_taxonomy(_prepare(data));prepared=engineer(prepared) if engineer else prepared;train,test=temporal_project_split(prepared,2001,end,test_end);train,test,_=_build_temporal_delay_priors(train,test);cohort=_production_cost_evaluation_rows(test).copy();ids=_select_aft_calibration_projects(cohort);cohort[CALIBRATION_GATE_FEATURE]=cohort['canonical_project_id'].astype('string').isin(ids);cohort=assign_project_balanced_weights(cohort)
        if cohort['canonical_project_id'].nunique()!=projects or len(cohort)!=snapshots: raise RuntimeError('Comparison cohort changed')
        ctx={'training_end':end,'test_start':test_start,'test_end':test_end,'train':train,'cohort':cohort,'cost_model':cm,'delay_model':dm,'production_cost':np.asarray(cm.predict(cohort),float),'production_delay':np.maximum(0,np.asarray(dm.predict(cohort),float))}
    if before!=_hash_prod(): raise AssertionError('Tracked production artifacts changed')
    return ctx

def persist(exp_id,name,ctx,experiment_delay,details,output):
    c=ctx['cohort'];pc=ctx['production_cost'];pdly=ctx['production_delay'];ed=np.maximum(0,np.asarray(experiment_delay,float));pcm=_metric(c,'actual_cost_overrun_percentage',pc);pdm=_metric(c,'actual_delay_days',pdly);edm=_metric(c,'actual_delay_days',ed);dg=_gain(pdm,edm);verdict='PROMOTION CANDIDATE' if dg>0 else 'DO NOT PROMOTE';result={'experiment_id':exp_id,'experiment_name':name,'scope':'delay','training_start':2001,'training_end':ctx['training_end'],'test_start':ctx['test_start'],'test_end':ctx['test_end'],'production_cost_mae':pcm,'experiment_cost_mae':pcm,'cost_improvement_percentage':0.0,'production_delay_mae':pdm,'experiment_delay_mae':edm,'delay_improvement_percentage':round(dg,6),'comparison_test_projects':int(c['canonical_project_id'].nunique()),'comparison_test_snapshots':len(c),'cost_predictions_identical':True,'holdout_used_for_selection':False,'promotion_allowed':False,'execution_verdict':'EXECUTION VALID','scientific_verdict':verdict,'details':_json_safe(details)};p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(result),indent=2,allow_nan=False)+'\n');mark=exp_id.replace('_','').upper();print(f'{mark}_2001_{ctx["training_end"]}_PRODUCTION_COST_MAE={pcm:.6f}');print(f'{mark}_2001_{ctx["training_end"]}_EXPERIMENT_COST_MAE={pcm:.6f}');print(f'{mark}_2001_{ctx["training_end"]}_PRODUCTION_DELAY_MAE={pdm:.6f}');print(f'{mark}_2001_{ctx["training_end"]}_EXPERIMENT_DELAY_MAE={edm:.6f}');print(f'{mark}_2001_{ctx["training_end"]}_DELAY_IMPROVEMENT_PERCENT={dg:.6f}');print(f'{mark}_SCIENTIFIC_VERDICT={verdict}');return result
