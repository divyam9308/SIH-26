"""Exp103: project-type x size x lifecycle shrunk Cost residual priors."""
from __future__ import annotations
import argparse,re,numpy as np,pandas as pd
from backend.app.ml.experiments.post_u1_cost_common import current_cost_oof,fit_residual_booster,persist,prepare_context,window_contract
EXPERIMENT_ID='exp_103';EXPERIMENT_NAME='Project-type x size x stage Cost residual priors';EXPERIMENT_SCOPE='cost';EXPERIMENT_SEQUENCE=103
PATTERNS=[('metro',r'\bmetro\b'),('rail',r'rail|railway'),('road',r'road|highway|expressway'),('bridge',r'bridge'),('tunnel',r'tunnel'),('hydro',r'hydro'),('power',r'thermal|solar|wind|power'),('transmission',r'transmission|grid'),('port',r'port|harbour'),('airport',r'airport'),('water',r'irrigation|water|dam'),('pipeline',r'pipeline')]
def _context(frame):
    x=frame.copy();name=x.get('project_name',pd.Series('',index=x.index)).fillna('').astype(str).str.lower();typ=np.full(len(x),'other',object)
    for label,pat in PATTERNS:typ=np.where(name.str.contains(pat,regex=True),label,typ)
    x['exp103_type']=typ;dr=pd.to_numeric(x['duration_ratio'],errors='coerce');x['exp103_stage']=pd.cut(dr,[-np.inf,.25,.6,1.0,np.inf],labels=['early','mid','late','very_late']).astype('string').fillna('unknown');x['exp103_key']=x['exp103_type'].astype(str)+'|'+x['project_size_category'].astype(str)+'|'+x['exp103_stage'].astype(str);return x
def _prior(hist,key='exp103_key'):
    h=hist.copy();h['_wr']=pd.to_numeric(h['residual'],errors='coerce')*pd.to_numeric(h['sample_weight'],errors='coerce').fillna(0);g=h.groupby(key,dropna=False).agg(s=('_wr','sum'),w=('sample_weight','sum'),n=('canonical_project_id','nunique'));raw=g.s/g.w.replace(0,np.nan);global_mean=float(h['_wr'].sum()/max(h['sample_weight'].sum(),1e-9));return (raw*g.n+global_mean*12)/(g.n+12)
def fit_experiment(training_end,output):
    window_contract(training_end);ctx=prepare_context(training_end);oof=_context(current_cost_oof(ctx['train'],ctx['cost_model']));score=_context(ctx['cohort']);score['production_prediction']=ctx['production_cost'];yy=pd.to_numeric(oof['oof_year'],errors='coerce');oof['exp103_prior']=np.nan
    for year in sorted(int(v) for v in yy.dropna().unique()):
        hist=oof.loc[yy<year];val=oof.loc[yy==year]
        if not hist.empty:oof.loc[yy==year,'exp103_prior']=val['exp103_key'].map(_prior(hist)).to_numpy()
    score['exp103_prior']=score['exp103_key'].map(_prior(oof));features=['production_prediction','exp103_prior','duration_ratio','approved_cost_cr','cost_escalation_percentage','expenditure_ratio'];corr,meta=fit_residual_booster(oof,score,features,10301);meta['semantic_types']=[x[0] for x in PATTERNS]+['other'];meta['shrinkage_projects']=12;return persist(EXPERIMENT_ID,EXPERIMENT_NAME,ctx,ctx['production_cost']+corr,meta,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2019,2021],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
