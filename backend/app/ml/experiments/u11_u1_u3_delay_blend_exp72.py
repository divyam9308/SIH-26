"""Experiment 72: training-only OOF convex blend of the U1 and U3 Delay challengers.

U1 is a bounded nonlinear residual booster. U3 is a large-error risk gate plus
residual specialist. The final blend weight is selected only from forward meta-OOF
training evidence, then frozen before the future holdout is scored. Cost is exact
current Exp61 production.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import delay_oof_frame,production_comparison,run_cli,weighted_quantile

EXPERIMENT_ID='exp_72';EXPERIMENT_SEQUENCE=72;MARKER='EXP72'
EXPERIMENT_NAME='U1 + U3 training-only OOF Delay blend';EXPERIMENT_SCOPE='delay'
CHANGED_DIMENSION='forward_meta_oof_convex_blend_of_u1_u3_delay_predictions'
FEATURES=['production_prediction','cost_escalation_percentage','schedule_slippage_days','duration_ratio','expenditure_ratio','progress_deviation','approved_cost_cr','exp58_delay_hier_prior','exp58_group_support']
WEIGHT_GRID=np.linspace(0.0,1.0,21)

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

def _u3(train,score,seed):
    cols,x,xs,med=_design(train,score);r=pd.to_numeric(train['residual'],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float);w=pd.to_numeric(train['sample_weight'],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float);thr=float(weighted_quantile(np.abs(r),w,.75));label=(np.abs(r)>=thr).astype(int)
    if len(np.unique(label))<2:gate=np.full(len(xs),bool(label[0]))
    else:
        clf=RandomForestClassifier(n_estimators=220,max_depth=5,min_samples_leaf=25,class_weight='balanced_subsample',random_state=seed,n_jobs=2);clf.fit(x,label,sample_weight=w);gate=clf.predict_proba(xs)[:,1]>=.5
    sm=label==1
    if int(sm.sum())<40:sm=np.ones(len(label),dtype=bool)
    reg=LGBMRegressor(n_estimators=160,learning_rate=.025,max_depth=3,num_leaves=12,min_child_samples=50,reg_alpha=5,reg_lambda=25,random_state=seed+100,verbosity=-1);reg.fit(x.loc[sm],r[sm],sample_weight=w[sm]);corr=np.asarray(reg.predict(xs),float);cap=max(float(weighted_quantile(np.abs(r[sm]),w[sm],.90)),1e-9);return np.where(gate,np.clip(corr,-cap,cap),0.0),{'features':cols,'training_medians':med,'large_error_q75':thr,'routed_rows':int(gate.sum()),'cap_q90':cap}

def _weighted_mae(actual,pred,weight):
    a=np.asarray(actual,float);p=np.asarray(pred,float);w=np.asarray(weight,float);m=np.isfinite(a)&np.isfinite(p)&np.isfinite(w)&(w>=0)
    return float(np.average(np.abs(a[m]-p[m]),weights=w[m])) if m.any() and float(w[m].sum())>0 else float(np.mean(np.abs(a[m]-p[m])))

def select_u1_weight(oof):
    years=sorted(int(x) for x in pd.to_numeric(oof['oof_year'],errors='coerce').dropna().unique());parts=[];used=[]
    for year in years[1:]:
        fit=oof[pd.to_numeric(oof['oof_year'],errors='coerce')<year].copy();val=oof[pd.to_numeric(oof['oof_year'],errors='coerce')==year].copy()
        if len(fit)<80 or val.empty:continue
        c1,_=_u1(fit,val,7200+year);c3,_=_u3(fit,val,7300+year);base=pd.to_numeric(val['production_prediction'],errors='coerce').to_numpy(float);p1=np.maximum(0,base+c1);p3=np.maximum(0,base+c3);parts.append(pd.DataFrame({'actual':pd.to_numeric(val['actual_delay_days'],errors='coerce').to_numpy(float),'weight':pd.to_numeric(val['sample_weight'],errors='coerce').to_numpy(float),'u1':p1,'u3':p3}));used.append(year)
    if not parts:return .5,{'selection_years':[],'fallback_weight':.5,'reason':'insufficient forward meta-OOF rows'}
    x=pd.concat(parts,ignore_index=True);scores=[]
    for wu1 in WEIGHT_GRID:
        pred=float(wu1)*x['u1'].to_numpy(float)+(1-float(wu1))*x['u3'].to_numpy(float);scores.append((float(wu1),_weighted_mae(x['actual'],pred,x['weight'])))
    best=min(scores,key=lambda z:(z[1],abs(z[0]-.5)));return best[0],{'selection_years':used,'selected_u1_weight':best[0],'selected_u3_weight':1-best[0],'meta_oof_mae':best[1],'weight_grid':[float(x) for x in WEIGHT_GRID],'all_scores':[{'u1_weight':w,'mae':m} for w,m in scores]}

def fit_experiment(*,data,production_bundle,training_start,training_end,test_end,**kwargs):
    _,_,c,pc,pdly=production_comparison(data,production_bundle,training_start,training_end,test_end);oof=delay_oof_frame(data,production_bundle,training_start,training_end,test_end);wu1,selection=select_u1_weight(oof);score=c.copy();score['production_prediction']=pdly;c1,d1=_u1(oof,score,7201);c3,d3=_u3(oof,score,7301);p1=np.maximum(0,pdly+c1);p3=np.maximum(0,pdly+c3);edly=np.maximum(0,wu1*p1+(1-wu1)*p3)
    return _persist(EXPERIMENT_ID,EXPERIMENT_NAME,EXPERIMENT_SCOPE,CHANGED_DIMENSION,c,pc,pc.copy(),pdly,edly,{'baseline':'current production (Exp61)','selection':selection,'u1_final':d1,'u3_final':d3,'cost_predictions_identical':True,'holdout_used_for_weight_selection':False,'blend_formula':'w*U1 + (1-w)*U3'})

if __name__=='__main__':run_cli(sys.modules[__name__])
