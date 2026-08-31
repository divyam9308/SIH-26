"""Experiment 89: lifecycle/evidence-adaptive scaling of the existing U1 correction."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_delay_common import _mae,current_delay_oof,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_89';EXPERIMENT_NAME='Lifecycle-adaptive U1 correction scale';EXPERIMENT_SCOPE='delay';EXPERIMENT_SEQUENCE=89
GRID=(0.0,.25,.5,.75,1.0)

def _stage(v): return '<NA>' if pd.isna(v) else str(v)
def _tier(v):
    try: x=float(v)
    except Exception: return 'missing'
    if not np.isfinite(x): return 'missing'
    return 'low' if x<5 else 'mid' if x<20 else 'high'
def _key(frame):
    stage=frame.get('lifecycle_stage',pd.Series(pd.NA,index=frame.index));support=frame.get('exp58_group_support',pd.Series(np.nan,index=frame.index));return [(_stage(s),_tier(u)) for s,u in zip(stage,support)]
def _best_scale(frame):
    base=pd.to_numeric(frame['base_prediction'],errors='coerce').to_numpy(float);corr=pd.to_numeric(frame['u1_correction'],errors='coerce').to_numpy(float);y=pd.to_numeric(frame['actual_delay_days'],errors='coerce').to_numpy(float);w=pd.to_numeric(frame['sample_weight'],errors='coerce').to_numpy(float);return min(GRID,key=lambda a:(_mae(y,np.maximum(0,base+a*corr),w),a))

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_delay_oof(ctx['train'],ctx['delay_model']);global_scale=float(_best_scale(oof));keys=_key(oof);oof=oof.copy();oof['_exp89_key']=keys;mapping={};supports={}
    for k,g in oof.groupby('_exp89_key'):
        n=int(g['canonical_project_id'].nunique());supports[str(k)]=n;mapping[k]=float(_best_scale(g)) if n>=30 else global_scale
    score=ctx['cohort'].copy();base=np.maximum(0,np.asarray(ctx['delay_model'].base_model.predict(score),float));corr=ctx['production_delay']-base;scales=np.asarray([mapping.get(k,global_scale) for k in _key(score)],float);pred=np.maximum(0,base+scales*corr);details={'global_u1_scale':global_scale,'scale_grid':list(GRID),'group_scales':{str(k):v for k,v in mapping.items()},'group_project_support':supports,'minimum_projects_for_group_scale':30,'maximum_scale':1.0,'holdout_used_for_scale_selection':False};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
