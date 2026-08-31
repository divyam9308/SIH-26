"""Experiment 82: lifecycle-adaptive Cost calibration strength."""
from __future__ import annotations
import argparse,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import _mae,_wq,current_cost_oof,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_82';EXPERIMENT_NAME='Lifecycle-adaptive Cost calibration strength';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=82

def _stage(v): return '<NA>' if pd.isna(v) else str(v)
def _table(fit,strength):
    p=pd.to_numeric(fit['production_prediction'],errors='coerce');edges=np.unique(np.quantile(p.dropna(),np.linspace(0,1,6)).astype(float));edges=np.array([-np.inf,np.inf]) if len(edges)<3 else np.r_[-np.inf,edges[1:-1],np.inf]
    x=fit.copy();x['_bin']=np.digitize(p.to_numpy(float),edges[1:-1]);r=pd.to_numeric(x['residual'],errors='coerce').fillna(0).to_numpy(float);w=pd.to_numeric(x['sample_weight'],errors='coerce').fillna(0).to_numpy(float);global_med=_wq(r,w,.5);bins={};cells={}
    for b,g in x.groupby('_bin'):
        ww=pd.to_numeric(g['sample_weight'],errors='coerce').fillna(0).to_numpy(float);rr=pd.to_numeric(g['residual'],errors='coerce').fillna(0).to_numpy(float);s=float(ww.sum());bins[int(b)]=(s*_wq(rr,ww,.5)+strength*global_med)/(s+strength)
    for (st,b),g in x.groupby(['lifecycle_stage','_bin'],dropna=False):
        ww=pd.to_numeric(g['sample_weight'],errors='coerce').fillna(0).to_numpy(float);rr=pd.to_numeric(g['residual'],errors='coerce').fillna(0).to_numpy(float);s=float(ww.sum());parent=bins.get(int(b),global_med);cells[(_stage(st),int(b))]=(s*_wq(rr,ww,.5)+strength*parent)/(s+strength)
    return edges,global_med,bins,cells

def _lookup(score,t):
    edges,g,bins,cells=t;p=pd.to_numeric(score['production_prediction'],errors='coerce').fillna(0).to_numpy(float);bb=np.digitize(p,edges[1:-1]);st=score.get('lifecycle_stage',pd.Series(pd.NA,index=score.index));return np.asarray([cells.get((_stage(s),int(b)),bins.get(int(b),g)) for s,b in zip(st,bb)],float)

def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=current_cost_oof(ctx['train'],ctx['cost_model']);years=sorted(int(x) for x in pd.to_numeric(oof['oof_year'],errors='coerce').dropna().unique());best=(float('inf'),40.0,0.0)
    for strength in (20.0,40.0,80.0,160.0):
      for scale in (0.0,.25,.5,.75,1.0):
        vals=[];weights=[]
        for year in years[1:]:
            yy=pd.to_numeric(oof['oof_year'],errors='coerce');fit=oof.loc[yy<year].copy();val=oof.loc[yy==year].copy()
            if len(fit)<80 or val.empty: continue
            corr=_lookup(val,_table(fit,strength));pred=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float)+scale*corr;y=pd.to_numeric(val['actual_cost_overrun_percentage'],errors='coerce').to_numpy(float);w=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);vals.append(_mae(y,pred,w));weights.append(max(float(w.sum()),1e-9))
        if vals:
            m=float(np.average(vals,weights=weights));best=min(best,(m,strength,scale))
    _,strength,scale=best;score=ctx['cohort'].copy();score['production_prediction']=ctx['production_cost'];corr=_lookup(score,_table(oof,strength));pred=ctx['production_cost']+scale*corr;details={'selected_strength':strength,'selected_scale':scale,'candidate_strengths':[20,40,80,160],'meta_oof_years':years[1:],'estimand':'current-production residual by lifecycle-stage and prediction bin'};return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,pred,details,output)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--end',type=int,required=True,choices=[2019,2021]);ap.add_argument('--output',required=True);a=ap.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__': main()
