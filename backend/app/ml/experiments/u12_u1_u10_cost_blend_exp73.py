"""Experiment 73: production-anchored training-only OOF Cost blend of U1 and U10.

The blend searches only forward meta-OOF training evidence. Production is constrained
to at least 50% weight; U1 and U10 may only make conservative complementary
corrections. The future holdout never selects weights. Delay is exact production.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from backend.app.ml.experiments.exp35_aft_residual_combo import _corrections
from backend.app.ml.experiments.nextgen_common import _compare,_family,_persist,_prepare,normalize_taxonomy,shrunk_calibration
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.experiments.post61_common import cost_oof_frame,production_comparison,run_cli,weighted_quantile
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import PRODUCTION_COST_SEED

EXPERIMENT_ID='exp_73';EXPERIMENT_SEQUENCE=73;MARKER='EXP73'
EXPERIMENT_NAME='Production + U1 + U10 training-only OOF Cost blend';EXPERIMENT_SCOPE='cost'
CHANGED_DIMENSION='production_anchored_forward_meta_oof_blend_u1_u10_cost'
FEATURES=['production_prediction','cost_escalation_percentage','schedule_slippage_days','duration_ratio','expenditure_ratio','progress_deviation','approved_cost_cr','exp58_delay_hier_prior','exp58_group_support']
WINDOWS=(3,6,12,24,36);SIGNALS=('cost_escalation_percentage','expenditure_ratio')
TREND_FEATURES=[f'exp73_{s}_slope_{w}m' for s in SIGNALS for w in WINDOWS]+[f'exp73_{s}_curvature_{a}_{b}' for s in SIGNALS for a,b in ((3,12),(6,24),(12,36))]
KEYS=['canonical_project_id','snapshot_date']

def _design(train,score):
    cols=[c for c in FEATURES if c in train.columns and c in score.columns]
    if 'production_prediction' not in cols:raise AssertionError('production prediction missing')
    a={};b={};med={}
    for c in cols:
        tr=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan);sc=pd.to_numeric(score[c],errors='coerce').replace([np.inf,-np.inf],np.nan);finite=tr[np.isfinite(tr.to_numpy(float))];m=float(finite.median()) if len(finite) else 0.0;med[c]=m;a[c]=tr.fillna(m);b[c]=sc.fillna(m)
    return cols,pd.DataFrame(a,index=train.index),pd.DataFrame(b,index=score.index),med

def _u1(train,score,seed):
    cols,x,xs,med=_design(train,score);r=pd.to_numeric(train['residual'],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    model=LGBMRegressor(n_estimators=180,learning_rate=.025,max_depth=3,num_leaves=12,min_child_samples=80,subsample=.85,colsample_bytree=.85,reg_alpha=5,reg_lambda=25,random_state=seed,verbosity=-1);model.fit(x,r,sample_weight=w);corr=np.asarray(model.predict(xs),float);cap=max(float(weighted_quantile(np.abs(r),w,.90)),1e-9);return np.clip(corr,-cap,cap),{'features':cols,'training_medians':med,'cap_q90':cap}

def _median_pairwise_slope(days,values):
    x=np.asarray(days,float);y=np.asarray(values,float);m=np.isfinite(x)&np.isfinite(y);x=x[m];y=y[m]
    if len(x)<2:return np.nan
    i,j=np.triu_indices(len(x),1);dx=x[j]-x[i];good=dx>0
    if not good.any():return np.nan
    return float(np.median((y[j][good]-y[i][good])/dx[good]*30.4375))

def enrich_robust_trends(frame):
    out=frame.copy();out['snapshot_date']=pd.to_datetime(out['snapshot_date'],errors='coerce');result={n:pd.Series(np.nan,index=out.index,dtype=float) for n in TREND_FEATURES};ordered=out.sort_values(['canonical_project_id','snapshot_date'],kind='mergesort')
    for _,part in ordered.groupby('canonical_project_id',sort=False):
        idx=part.index.to_numpy();days=part['snapshot_date'].astype('int64').to_numpy(float)/86400e9
        for signal in SIGNALS:
            values=pd.to_numeric(part.get(signal),errors='coerce').to_numpy(float);slopes={w:np.full(len(part),np.nan,float) for w in WINDOWS}
            for pos in range(len(part)):
                for w in WINDOWS:
                    lo=int(np.searchsorted(days,days[pos]-w*30.4375,side='left'));slopes[w][pos]=_median_pairwise_slope(days[lo:pos+1],values[lo:pos+1])
            for w in WINDOWS:result[f'exp73_{signal}_slope_{w}m'].loc[idx]=slopes[w]
            for a,b in ((3,12),(6,24),(12,36)):result[f'exp73_{signal}_curvature_{a}_{b}'].loc[idx]=slopes[a]-slopes[b]
    for n,s in result.items():out[n]=s
    return out

def _u10_raw_oof(train,features,family):
    chunks=[]
    for fit,val,year in _rolling_folds(train):
        model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],fit,features,'actual_cost_overrun_percentage');raw=np.asarray(model.predict(val[features]),float);part=val.copy();part['prediction']=raw;part['residual']=pd.to_numeric(part['actual_cost_overrun_percentage'],errors='coerce')-raw;part['oof_year']=int(year);chunks.append(part)
    if len(chunks)<2:raise ValueError('U10 blend requires at least two rolling OOF folds')
    return pd.concat(chunks,ignore_index=True)

def _crossfit_u10(raw_oof):
    years=sorted(int(x) for x in pd.to_numeric(raw_oof['oof_year'],errors='coerce').dropna().unique());parts=[]
    for year in years[1:]:
        hist=raw_oof[pd.to_numeric(raw_oof['oof_year'],errors='coerce')<year].copy();val=raw_oof[pd.to_numeric(raw_oof['oof_year'],errors='coerce')==year].copy()
        if hist.empty or val.empty:continue
        cal=shrunk_calibration(hist,40.0);raw=pd.to_numeric(val['prediction'],errors='coerce').to_numpy(float);pred=raw+_corrections(val,raw,cal);x=val[KEYS+['oof_year']].copy();x['u10_prediction']=pred;parts.append(x)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=KEYS+['oof_year','u10_prediction'])

def _weighted_mae(actual,pred,weight):
    a=np.asarray(actual,float);p=np.asarray(pred,float);w=np.asarray(weight,float);m=np.isfinite(a)&np.isfinite(p)&np.isfinite(w)&(w>=0)
    return float(np.average(np.abs(a[m]-p[m]),weights=w[m])) if m.any() and float(w[m].sum())>0 else float(np.mean(np.abs(a[m]-p[m])))

def select_weights(prod_oof,u10_cf):
    years=sorted(int(x) for x in pd.to_numeric(prod_oof['oof_year'],errors='coerce').dropna().unique());parts=[];used=[]
    for year in years[1:]:
        fit=prod_oof[pd.to_numeric(prod_oof['oof_year'],errors='coerce')<year].copy();val=prod_oof[pd.to_numeric(prod_oof['oof_year'],errors='coerce')==year].copy()
        if len(fit)<80 or val.empty:continue
        corr,_=_u1(fit,val,7300+year);base=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float);x=val[KEYS+['oof_year']].copy();x['actual']=pd.to_numeric(val['actual_cost_overrun_percentage'],errors='coerce').to_numpy(float);x['weight']=pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float);x['production']=base;x['u1']=base+corr;parts.append(x);used.append(year)
    if not parts:return (.5,.25,.25),{'selection_years':[],'fallback_weights':{'production':.5,'u1':.25,'u10':.25},'reason':'insufficient forward meta-OOF rows'}
    x=pd.concat(parts,ignore_index=True).merge(u10_cf,on=KEYS+['oof_year'],how='inner',validate='one_to_one')
    if x.empty:return (.5,.25,.25),{'selection_years':[],'fallback_weights':{'production':.5,'u1':.25,'u10':.25},'reason':'no aligned U10 forward OOF rows'}
    scores=[]
    for ip in range(10,21):
        for iu1 in range(0,21-ip):
            iu10=20-ip-iu1;wp,wu1,wu10=ip/20.0,iu1/20.0,iu10/20.0;pred=wp*x['production'].to_numpy(float)+wu1*x['u1'].to_numpy(float)+wu10*x['u10_prediction'].to_numpy(float);scores.append((wp,wu1,wu10,_weighted_mae(x['actual'],pred,x['weight'])))
    best=min(scores,key=lambda z:(z[3],-z[0],abs(z[1]-z[2])));return best[:3],{'selection_years':used,'aligned_meta_oof_rows':int(len(x)),'selected_weights':{'production':best[0],'u1':best[1],'u10':best[2]},'meta_oof_mae':best[3],'production_min_weight':.5,'grid_step':.05}

def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,c,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end);frame=enrich_robust_trends(normalize_taxonomy(_prepare(data)));train,test=temporal_project_split(frame,training_start,training_end,test_end);trend_rows=test[KEYS+TREND_FEATURES].drop_duplicates(KEYS);c=c.copy();c['_order']=np.arange(len(c));c=c.merge(trend_rows,on=KEYS,how='left',sort=False,validate='one_to_one').sort_values('_order').drop(columns='_order').reset_index(drop=True)
    cm=production_bundle['cost'];family=_family(cm);features=list(dict.fromkeys(list(cm.features)+TREND_FEATURES));prod_oof=cost_oof_frame(data,production_bundle,training_start,training_end,test_end);raw_oof=_u10_raw_oof(train,features,family);u10_cf=_crossfit_u10(raw_oof);(wp,wu1,wu10),selection=select_weights(prod_oof,u10_cf)
    score=c.copy();score['production_prediction']=pc;u1_corr,u1_details=_u1(prod_oof,score,7301);pu1=pc+u1_corr;cal=shrunk_calibration(raw_oof,40.0);model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],train,features,'actual_cost_overrun_percentage');raw=np.asarray(model.predict(c[features]),float);pu10=raw+_corrections(c,raw,cal);ec=wp*pc+wu1*pu1+wu10*pu10
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,ec,pdly,pdly.copy(),{'baseline':'current production (Exp61)','selection':selection,'u1_final':u1_details,'u10_features':TREND_FEATURES,'u10_calibration_strength':40.0,'delay_predictions_identical':True,'holdout_used_for_weight_selection':False,'blend_formula':'w_prod*production + w_u1*U1 + w_u10*U10','production_weight_floor':.5})

if __name__=='__main__':run_cli(sys.modules[__name__])
