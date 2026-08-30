"""Shared harness for isolated challengers evaluated against current Exp61 production."""
from __future__ import annotations
import argparse,json,tempfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from backend.app.ml.experiments.exp35_aft_residual_combo import _aft_remaining_prediction,_corrections,_delay_from_remaining,_fit_aft_family_models,_remaining_frame
from backend.app.ml.experiments.nextgen_common import _compare,_family,_hash_prod,_prepare,normalize_taxonomy
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_json_safe,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors,train_window_with_promoted_cost_and_delay

def prepared_split(data,start,end,test_end):
    frame=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(frame,start,end,test_end);return (*_build_temporal_delay_priors(train,test)[:2],)

def production_comparison(data,bundle,start,end,test_end):
    train,test=prepared_split(data,start,end,test_end);cohort=_compare(test);cost=np.asarray(bundle['cost'].predict(cohort),float);delay=np.maximum(0.0,np.asarray(bundle['delay'].predict(cohort),float));return train,test,cohort,cost,delay

def cost_oof_frame(data,bundle,start,end,test_end):
    train,_=temporal_project_split(normalize_taxonomy(_prepare(data)),start,end,test_end);model=bundle['cost'];features=list(model.features);family=_family(model);chunks=[]
    for fit,val,year in _rolling_folds(train):
        pipe=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],fit,features,'actual_cost_overrun_percentage');raw=np.asarray(pipe.predict(val[features]),float);pred=raw+_corrections(val,raw,model.calibration);part=val.copy();part['production_prediction']=pred;part['residual']=pd.to_numeric(part['actual_cost_overrun_percentage'],errors='coerce')-pred;part['oof_year']=int(year);chunks.append(part)
    if len(chunks)<2: raise ValueError('Cost residual challenger requires at least two rolling OOF folds')
    return pd.concat(chunks,ignore_index=True)

def delay_oof_frame(data,bundle,start,end,test_end):
    train,test=temporal_project_split(normalize_taxonomy(_prepare(data)),start,end,test_end);prior_train,_,_=_build_temporal_delay_priors(train,test.iloc[:0].copy());model=bundle['delay'];features=list(model.model_features);train_delay=_remaining_frame(prior_train);chunks=[]
    for fit,val,year in _rolling_folds(train_delay):
        models=_fit_aft_family_models(fit,features);remaining=_aft_remaining_prediction(models,model.weights,val,features);raw=_delay_from_remaining(val,remaining);pred=np.maximum(0.0,raw+_corrections(val,raw,model.calibration));part=val.copy();part['production_prediction']=pred;part['residual']=pd.to_numeric(part['actual_delay_days'],errors='coerce')-pred;part['oof_year']=int(year);chunks.append(part)
    if len(chunks)<2: raise ValueError('Delay residual challenger requires at least two rolling OOF folds')
    return pd.concat(chunks,ignore_index=True)

def weighted_quantile(values,weights,q):
    values=np.asarray(values,float);weights=np.asarray(weights,float);mask=np.isfinite(values)&np.isfinite(weights)&(weights>=0);values,weights=values[mask],weights[mask]
    if not len(values): return 0.0
    order=np.argsort(values);values,weights=values[order],weights[order];total=float(weights.sum())
    if total<=0:return float(np.quantile(values,q))
    return float(values[np.searchsorted(np.cumsum(weights),q*total,side='left')])

def run_cli(module):
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=2001);p.add_argument('--end',type=int,required=True);p.add_argument('--test-end',type=int,default=2025);p.add_argument('--output',required=True);a=p.parse_args()
    if a.start!=2001 or a.end not in (2019,2021) or a.test_end!=2025: raise ValueError('Post-Exp61 experiments support 2001-2019 and 2001-2021 through 2025 only')
    before=_hash_prod();data,identity=build_training_dataset()
    with tempfile.TemporaryDirectory(prefix=module.EXPERIMENT_ID+'-') as td:
        root=Path(td)/'production';receipt=train_window_with_promoted_cost_and_delay(a.start,a.end,a.test_end,data=data,identity=identity,artifact_root=root);target=root/f'{a.start}_{a.end}';bundle={'cost':joblib.load(target/'cost_model.pkl'),'delay':joblib.load(target/'delay_model.pkl'),'metadata':json.loads((target/'metadata.json').read_text())};fitted=module.fit_experiment(data=data,production_bundle=bundle,production_receipt=receipt,training_start=a.start,training_end=a.end,test_end=a.test_end)
    if before!=_hash_prod():raise AssertionError('Experiment modified tracked production artifacts')
    overall=fitted['overall_comparison']
    if a.end==2021 and (overall['comparison_test_projects']!=721 or overall['comparison_test_snapshots']!=11200):raise RuntimeError('Exp61 decision cohort changed')
    payload={'window':f'{a.start}_{a.end}','test_end':a.test_end,'assumed_production':'Exp61 (PR #96)','production':receipt,'experiment':fitted['experiment'],'overall_comparison':overall,'production_artifacts_untouched':True};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(_json_safe(payload),indent=2,allow_nan=False)+'\n')
    prefix=f'{module.MARKER}_{a.start}_{a.end}'
    for key in ('production_cost_mae','experiment_cost_mae','cost_improvement_percentage','production_delay_mae','experiment_delay_mae','delay_improvement_percentage'):print(prefix+'_'+key.upper()+'='+str(overall[key]))
    print(module.MARKER+'_EXECUTION_VERDICT='+overall['execution_verdict']);print(module.MARKER+'_SCIENTIFIC_VERDICT='+overall['scientific_verdict'])
