"""Route-and-stage hierarchical recalibration of the promoted Exp113 correction."""
from __future__ import annotations
import argparse
import numpy as np,pandas as pd
from backend.app.ml.experiments.post_exp113_component_common import build_component_oof_fold,component_context,component_oof,load_component_oof_dir
from backend.app.ml.experiments.post_exp113_delay_common import persist
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE
EXP_ID='exp_delay_route_stage_recalibration';NAME='Route-and-Stage Exp113 Recalibration';BETAS=(0.,.5,1.,1.5);LAMBDAS=(20.,40.,80.);MIN_SUPPORT=30.
def _stage(frame):
    if 'lifecycle_stage' in frame:
        s=frame['lifecycle_stage'].astype('string').fillna('missing');return s.replace({'<NA>':'missing'})
    ratio=pd.to_numeric(frame.get('duration_ratio'),errors='coerce');return pd.cut(ratio,[-np.inf,.5,.9,1.1,np.inf],labels=['early','mid','late','very_late']).astype('string').fillna('missing')
def _route(frame):
    gate=frame.get(CALIBRATION_GATE_FEATURE,pd.Series(False,index=frame.index)).fillna(False).astype(bool);planned=pd.to_datetime(frame.get('planned_completion_date'),errors='coerce');snapshot=pd.to_datetime(frame.get('snapshot_date'),errors='coerce');return pd.Series(np.where(gate,'aft',np.where(planned.notna() & snapshot.notna(),'inside_missing_aft','fallback')),index=frame.index)
def _decorate(frame):
    out=frame.copy();out['_stage']=_stage(out);out['_route']=_route(out);out['_cell']=out['_route'].astype(str)+'|'+out['_stage'].astype(str);return out
def _weighted_median(values,weights):
    order=np.argsort(values);v=values[order];w=weights[order]
    if not len(v) or float(w.sum())<=0:return 0.
    return float(v[np.searchsorted(np.cumsum(w),.5*w.sum(),side='left')])
def _fit_params(train,lam):
    train=_decorate(train);y=pd.to_numeric(train['actual_delay_days'],errors='coerce').to_numpy(float);u1=pd.to_numeric(train['production_u1'],errors='coerce').to_numpy(float);c=pd.to_numeric(train['exp113_correction'],errors='coerce').to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').fillna(0).to_numpy(float);global_bias=_weighted_median(y-(u1+c),w);route_bias={}
    for route,idx in train.groupby('_route').groups.items():
        ii=train.index.get_indexer(idx);route_bias[str(route)]=_weighted_median(y[ii]-(u1[ii]+c[ii]),w[ii])
    params={}
    for cell,idx in train.groupby('_cell').groups.items():
        ii=train.index.get_indexer(idx);support=float(w[ii].sum());route=str(cell).split('|',1)[0];best=None
        for beta in BETAS:
            residual=y[ii]-(u1[ii]+beta*c[ii]);raw=_weighted_median(residual,w[ii]);parent=route_bias.get(route,global_bias);shrink=support/(support+lam);bias=shrink*raw+(1-shrink)*parent;pred=np.maximum(0,u1[ii]+beta*c[ii]+bias);mae=float(np.average(np.abs(y[ii]-pred),weights=w[ii])) if support>0 else float('inf');choice=(mae,beta,bias,support)
            if best is None or choice[0]<best[0]:best=choice
        params[str(cell)]={'beta':float(best[1]),'bias':float(best[2]),'support':support,'route':route}
    return {'cells':params,'route_bias':route_bias,'global_bias':global_bias,'lambda':lam}
def _apply(frame,params):
    work=_decorate(frame);out=np.empty(len(work),dtype=float)
    for pos,(_,row) in enumerate(work.iterrows()):
        cell=str(row['_cell']);route=str(row['_route']);p=params['cells'].get(cell)
        if p is None or float(p['support'])<MIN_SUPPORT:beta=1.;bias=float(params['route_bias'].get(route,params['global_bias']))
        else:beta=float(p['beta']);bias=float(p['bias'])
        out[pos]=max(0.,float(row['production_u1'])+beta*float(row['exp113_correction'])+bias)
    return out
def _select(oof):
    years=sorted(int(v) for v in pd.to_numeric(oof['oof_year'],errors='coerce').dropna().unique());scores=[]
    for lam in LAMBDAS:
        folds=[]
        for year in years[1:]:
            yc=pd.to_numeric(oof['oof_year'],errors='coerce');fit=oof[yc<year];val=oof[yc==year]
            if len(fit)<100 or val.empty:continue
            pred=_apply(val,_fit_params(fit,lam));y=pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);folds.append((year,float(np.average(np.abs(y-pred),weights=w))))
        if folds:scores.append((float(np.mean([m for _,m in folds])),lam,folds))
    if not scores:raise ValueError('No valid forward meta-folds for route-stage recalibration')
    scores.sort(key=lambda x:x[0]);return float(scores[0][1]),{'candidate_lambdas':[{'lambda':x[1],'mean_meta_mae':x[0],'folds':x[2]} for x in scores]}
def run(output,oof=None):
    ctx=component_context();oof=component_oof(ctx) if oof is None else oof;lam,selection=_select(oof);params=_fit_params(oof,lam);score=ctx['cohort'].copy();score['production_u1']=ctx['production_u1'];score['exp113_correction']=ctx['exp113_correction'];prediction=_apply(score,params);details={'selected_lambda':lam,'selection':selection,'cell_count':len(params['cells']),'minimum_project_equivalent_support':MIN_SUPPORT,'holdout_used_for_selection':False,'full_holdout_retained':True,'oof_years':sorted(int(x) for x in pd.to_numeric(oof['oof_year']).unique())};return persist(EXP_ID,NAME,ctx,prediction,details,output)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='test-output/exp-delay-route-stage-recalibration/result.json');p.add_argument('--oof-dir');p.add_argument('--build-oof-year',type=int);p.add_argument('--oof-output');a=p.parse_args()
    if a.build_oof_year is not None:
        if not a.oof_output:p.error('--oof-output is required with --build-oof-year')
        build_component_oof_fold(a.build_oof_year,a.oof_output)
    else:run(a.output,load_component_oof_dir(a.oof_dir) if a.oof_dir else None)
