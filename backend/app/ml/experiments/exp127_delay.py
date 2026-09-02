"""Exp127: OOF error-regime mixture of experts."""
import argparse,pandas as pd
from sklearn.cluster import KMeans
from backend.app.ml.experiments.post_exp113_delay_common import *
EXPERIMENT_ID='exp127';NAME='OOF error-regime mixture of experts'
FEATS=['production_prediction','duration_ratio','schedule_slippage_days','physical_progress','progress_deviation','expenditure_ratio','cost_escalation_percentage']
def attach_clusters(train,score):
    cols,med,xt,xs=numeric_design(train,score,FEATS);mu=xt.mean();sd=xt.std().replace(0,1);m=KMeans(n_clusters=4,random_state=12700,n_init=10).fit((xt-mu)/sd);lab=m.predict((xs-mu)/sd);out=score.copy()
    for k in range(4):out[f'exp127_cluster_{k}']=(lab==k).astype(float)
    return out
def fit_experiment(end,output):
    c=prepare_context(end);o=production_oof(c);yc=pd.to_numeric(o['oof_year'],errors='coerce');parts=[]
    for y in sorted(int(x) for x in yc.dropna().unique())[1:]:
        f=o.loc[yc<y].copy();v=o.loc[yc==y].copy()
        if len(f)>=120 and not v.empty:parts.append(attach_clusters(f,v))
    meta=pd.concat(parts,ignore_index=True);s=c['cohort'].copy();s['production_prediction']=c['production_delay'];s=attach_clusters(o,s);features=FEATS+[f'exp127_cluster_{k}' for k in range(4)];corr,d=fit_residual(meta,s,features,12701);d['clusters']=4;return persist(EXPERIMENT_ID,NAME,c,c['production_delay']+corr,d,output)
def main():
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,choices=[2021,2022],required=True);p.add_argument('--output',required=True);a=p.parse_args();fit_experiment(a.end,a.output)
if __name__=='__main__':main()
# Comparison refreshed after shared timeout fix #173.
