"""Leakage-safe Exp113 component extraction for isolated experiments."""
from __future__ import annotations
import tempfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from backend.app.ml.experiments.nextgen_common import _prepare,normalize_taxonomy
from backend.app.ml.experiments.post_exp113_delay_common import forward_folds,prepare_context
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_exp105_exp113_baseline import train_window_with_promoted_cost_and_delay

def decompose(model,frame):
    u1=np.maximum(0,np.asarray(model.base_model.predict(frame),float));final=np.maximum(0,np.asarray(model.predict(frame),float));return u1,final-u1,final
def component_context():
    ctx=prepare_context(2021);u1,corr,final=decompose(ctx['delay_model'],ctx['cohort']);ctx['production_u1']=u1;ctx['exp113_correction']=corr;ctx['production_delay']=final;return ctx
def _training_context():
    data,identity=build_training_dataset();prepared=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(prepared,2001,2021,2025);train,_,_=_build_temporal_delay_priors(train,test);return {'full_data':data,'identity':identity,'train':train}
def _component_part(ctx,val,year):
    train_end=int(year)-1
    with tempfile.TemporaryDirectory(prefix=f'exp113-component-{year}-') as td:
        root=Path(td)/'models';train_window_with_promoted_cost_and_delay(2001,train_end,int(year),data=ctx['full_data'],identity=ctx['identity'],artifact_root=root,verify_frozen_reference=False);model=joblib.load(root/f'2001_{train_end}'/'delay_model.pkl');u1,corr,final=decompose(model,val)
    part=val.copy();part['production_u1']=u1;part['exp113_correction']=corr;part['production_prediction']=final;part['residual']=pd.to_numeric(part['actual_delay_days'],errors='coerce').to_numpy(float)-final;part['oof_year']=int(year)
    if CALIBRATION_GATE_FEATURE not in part: part[CALIBRATION_GATE_FEATURE]=False
    return part
def build_component_oof_fold(year,output,max_folds=6):
    ctx=_training_context();folds={int(y):val for _,val,y in forward_folds(ctx['train'],max_folds) if int(y)-1>=2005}
    if int(year) not in folds: raise ValueError(f'OOF year {year} not selected; expected {sorted(folds)}')
    part=_component_part(ctx,folds[int(year)],int(year));p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);joblib.dump(part,p,compress=3);print(f'DELAY_COMPONENT_OOF_FOLD_COMPLETED={year}; rows={len(part)}',flush=True);return p
def load_component_oof_dir(directory,expected=(2016,2017,2018,2019,2020,2021)):
    paths=sorted(Path(directory).glob('delay-oof-*.pkl'))
    if not paths: raise FileNotFoundError(f'No Delay OOF artifacts in {directory}')
    parts=[joblib.load(p) for p in paths];years=[int(pd.to_numeric(x['oof_year'],errors='raise').iloc[0]) for x in parts]
    if tuple(sorted(years))!=tuple(expected): raise ValueError(f'OOF years {sorted(years)} != {list(expected)}')
    if len(set(years))!=len(years): raise ValueError('Duplicate OOF year artifacts')
    return pd.concat(parts,ignore_index=True).sort_values(['oof_year','canonical_project_id','snapshot_date']).reset_index(drop=True)
def component_oof(ctx,max_folds=6):
    parts=[]
    for _,val,year in forward_folds(ctx['train'],max_folds):
        if int(year)-1<2005: continue
        parts.append(_component_part(ctx,val,int(year)));print(f'DELAY_COMPONENT_OOF_FOLD_COMPLETED={year}',flush=True)
    if len(parts)<3: raise ValueError('Need at least three strict forward Exp113 component folds')
    return pd.concat(parts,ignore_index=True)
