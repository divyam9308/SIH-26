"""Leakage-safe harness for isolated post-Exp105 Cost experiments."""
from __future__ import annotations
import json,tempfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from backend.app.ml.experiments.nextgen_common import _prepare,normalize_taxonomy
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights,build_training_dataset
from backend.app.ml.monthly_training import _json_safe,_regression_metrics,temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay

def metric(f,a,p): return float(_regression_metrics(f[a],p,f['sample_weight'],f['canonical_project_id'])['MAE'])
def regression_metrics(f,a,p): return _regression_metrics(f[a],p,f['sample_weight'],f['canonical_project_id'])
def numeric_design(train,score,features):
    cols=[c for c in features if c in train.columns and c in score.columns];med={};a={};b={}
    for c in cols:
        x=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan);y=pd.to_numeric(score[c],errors='coerce').replace([np.inf,-np.inf],np.nan);m=float(x.median()) if x.notna().any() else 0.0
        med[c]=m;a[c]=x.fillna(m);b[c]=y.fillna(m);a[f'{c}__missing']=x.isna().astype(float);b[f'{c}__missing']=y.isna().astype(float)
    return list(a),med,pd.DataFrame(a,index=train.index),pd.DataFrame(b,index=score.index)
def forward_folds(frame,max_folds=6):
    cy=pd.to_numeric(frame['completion_year'],errors='coerce');years=sorted(int(v) for v in cy.dropna().unique());out=[]
    for y in reversed(years[1:]):
        fit=frame.loc[cy<y].copy();val=frame.loc[cy==y].copy()
        if fit['canonical_project_id'].nunique()>=10 and val['canonical_project_id'].nunique()>=3: out.append((fit,val,y))
        if len(out)>=max_folds: break
    return list(reversed(out))
def _decompose(model,frame):
    base=np.asarray(model.base_model.predict(frame),float);final=np.asarray(model.predict(frame),float);return base,final-base,final
def _training_context(training_end=2021,test_end=2025):
    data,identity=build_training_dataset();prepared=normalize_taxonomy(_prepare(data));train,_=temporal_project_split(prepared,2001,training_end,test_end);return {'data':data,'identity':identity,'train':train}
def prepare_context(training_end=2021,test_end=2025):
    data,identity=build_training_dataset();prepared=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(prepared,2001,training_end,test_end);cohort=assign_project_balanced_weights(_production_cost_evaluation_rows(test).copy())
    with tempfile.TemporaryDirectory(prefix=f'post-exp105-{training_end}-') as td:
        root=Path(td)/'models';train_window_with_promoted_cost_and_delay(2001,training_end,test_end,data=data,identity=identity,artifact_root=root,verify_frozen_reference=(training_end==2021 and test_end==2025));model=joblib.load(root/f'2001_{training_end}'/'cost_model.pkl');base,corr,prod=_decompose(model,cohort)
    return {'training_end':training_end,'test_end':test_end,'data':data,'identity':identity,'train':train,'cohort':cohort,'production_model':model,'production_base':base,'production_correction':corr,'production_prediction':prod}
def _oof_part(ctx,val,year):
    train_end=int(year)-1
    with tempfile.TemporaryDirectory(prefix=f'exp105-oof-{year}-') as td:
        root=Path(td)/'models';train_window_with_promoted_cost_and_delay(2001,train_end,int(year),data=ctx['data'],identity=ctx['identity'],artifact_root=root,verify_frozen_reference=False);model=joblib.load(root/f'2001_{train_end}'/'cost_model.pkl');base,corr,pred=_decompose(model,val)
    part=val.copy();part['production_base']=base;part['exp105_correction']=corr;part['production_prediction']=pred;part['residual']=pd.to_numeric(part['actual_cost_overrun_percentage'],errors='coerce').to_numpy(float)-pred;part['oof_year']=int(year);return part
def build_oof_fold(year,output,max_folds=6):
    ctx=_training_context();folds={int(y):val for _,val,y in forward_folds(ctx['train'],max_folds) if int(y)-1>=2005}
    if int(year) not in folds: raise ValueError(f'OOF year {year} not selected; expected {sorted(folds)}')
    part=_oof_part(ctx,folds[int(year)],int(year));p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);joblib.dump(part,p,compress=3);print(f'COST_PRODUCTION_OOF_FOLD_COMPLETED={year}; rows={len(part)}',flush=True);return p
def load_oof_dir(directory,expected=(2016,2017,2018,2019,2020,2021)):
    paths=sorted(Path(directory).glob('cost-oof-*.pkl'))
    if not paths: raise FileNotFoundError(f'No cost OOF artifacts in {directory}')
    parts=[joblib.load(p) for p in paths];years=[int(pd.to_numeric(x['oof_year'],errors='raise').iloc[0]) for x in parts]
    if tuple(sorted(years))!=tuple(expected): raise ValueError(f'OOF years {sorted(years)} != {list(expected)}')
    if len(set(years))!=len(years): raise ValueError('Duplicate OOF year artifacts')
    return pd.concat(parts,ignore_index=True).sort_values(['oof_year','canonical_project_id','snapshot_date']).reset_index(drop=True)
def production_oof(ctx,max_folds=6):
    parts=[]
    for _,val,year in forward_folds(ctx['train'],max_folds):
        if int(year)-1<2005: continue
        parts.append(_oof_part(ctx,val,int(year)));print(f'COST_PRODUCTION_OOF_FOLD_COMPLETED={year}',flush=True)
    if len(parts)<3: raise ValueError('Need at least three strict forward Cost production OOF folds')
    return pd.concat(parts,ignore_index=True)
def persist(exp_id,name,ctx,prediction,details,output):
    c=ctx['cohort'];base=np.asarray(ctx['production_prediction'],float);cand=np.asarray(prediction,float);bm=regression_metrics(c,'actual_cost_overrun_percentage',base);cm=regression_metrics(c,'actual_cost_overrun_percentage',cand);verdict='PROMOTION CANDIDATE' if float(cm['MAE'])<float(bm['MAE']) else 'DO NOT PROMOTE'
    result={'experiment_id':exp_id,'experiment_name':name,'scope':'cost','training_start':2001,'training_end':ctx['training_end'],'test_start':ctx['training_end']+1,'test_end':ctx['test_end'],'comparison_test_projects':int(c['canonical_project_id'].nunique()),'comparison_test_snapshots':int(len(c)),'production':bm,'candidate':cm,'mae_delta':float(cm['MAE'])-float(bm['MAE']),'holdout_used_for_selection':False,'promotion_allowed':False,'scientific_verdict':verdict,'details':_json_safe(details)}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(_json_safe(result),indent=2,allow_nan=False)+'\n');print(f'{exp_id.upper()}_PRODUCTION_COST_MAE={float(bm["MAE"]):.6f}');print(f'{exp_id.upper()}_CANDIDATE_COST_MAE={float(cm["MAE"]):.6f}');print(f'{exp_id.upper()}_SCIENTIFIC_VERDICT={verdict}');return result
