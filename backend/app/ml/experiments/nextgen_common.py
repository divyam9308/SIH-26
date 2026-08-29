"""Shared isolated-experiment harness for Exp51-60 (2001-2021 only)."""
from __future__ import annotations
import argparse, hashlib, json, math, tempfile, uuid
from pathlib import Path
import joblib, numpy as np, pandas as pd
from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _aft_remaining_prediction,_corrections,_delay_aft_calibration_oof,_delay_from_remaining,
    _fit_aft_family_models,_fit_residual_calibration,_remaining_frame,
)
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds,enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import (
    assert_prediction_ledger_matches_cohort,build_prediction_ledger,write_experiment_prediction_ledger,
)
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights,build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_json_safe,_regression_metrics,_regressors,temporal_project_split
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,_production_cost_evaluation_rows,enrich_supervised_for_production,
)
from backend.app.ml.production_exp35_baseline import (
    AFTResidualDelayModel,CALIBRATION_GATE_FEATURE,_select_aft_calibration_projects,
    train_window_with_promoted_cost_and_delay,
)
ROOT=Path(__file__).resolve().parents[4]
EXPECTED=(721,11200,26.287,431.618); BOOTSTRAP=5000

def _hash_prod():
    root=ROOT/"models"/"monthly_lifecycle"; out={}
    if not root.exists(): return out
    for p in sorted(root.glob("*/*")):
        if p.is_file() and "experiments" not in p.parts:
            out[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out

def _prepare(data):
    f=enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    f["completion_year"]=pd.to_numeric(f["completion_year"],errors="coerce")
    f["snapshot_date"]=pd.to_datetime(f["snapshot_date"],errors="coerce")
    return f

def _compare(test):
    c=_production_cost_evaluation_rows(test).copy()
    ids=_select_aft_calibration_projects(c)
    c[CALIBRATION_GATE_FEATURE]=c["canonical_project_id"].astype("string").isin(ids)
    return assign_project_balanced_weights(c)

def _family(wrapper):
    name=wrapper.model.named_steps["model"].__class__.__name__.lower()
    if "extratrees" in name:return "extra_trees"
    if "lgbm" in name:return "lightgbm"
    if "xgb" in name:return "xgboost"
    raise ValueError(name)

def _metric(c,actual,p):
    return _regression_metrics(c[actual],p,c["sample_weight"],c["canonical_project_id"])["MAE"]

def _gain(a,b): return (float(a)-float(b))/float(a)*100 if float(a) else 0.0

def _cost_oof(train,features,family):
    chunks=[]
    for fit,val,year in _rolling_folds(train):
        m=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[family],fit,features,"actual_cost_overrun_percentage")
        p=np.asarray(m.predict(val[features]),float)
        x=val[["actual_cost_overrun_percentage","sample_weight","canonical_project_id","lifecycle_stage"]].copy()
        x["prediction"]=p; x["residual"]=pd.to_numeric(x["actual_cost_overrun_percentage"],errors="coerce")-p
        chunks.append(x)
    if len(chunks)<2: raise ValueError("need >=2 rolling folds")
    return pd.concat(chunks,ignore_index=True)

def _wm(v,w):
    v=np.asarray(v,float); w=np.asarray(w,float); mask=np.isfinite(v)&np.isfinite(w)&(w>=0);v=v[mask];w=w[mask]
    if not len(v):return 0.0
    o=np.argsort(v);v=v[o];w=w[o];cut=.5*w.sum()
    return float(np.median(v) if cut<=0 else v[np.searchsorted(np.cumsum(w),cut,side="left")])

def shrunk_calibration(oof,strength=40.0):
    pred=pd.to_numeric(oof["prediction"],errors="coerce"); edges=np.unique(np.quantile(pred,np.linspace(0,1,6)).astype(float))
    if len(edges)<3: edges=np.array([-np.inf,float(np.median(pred)),np.inf])
    else: edges[0]=-np.inf;edges[-1]=np.inf
    x=oof.copy();x["bin"]=np.digitize(pred.to_numpy(float),edges[1:-1]);g=_wm(x["residual"],x["sample_weight"])
    bm={}; sb={}
    for b,p in x.groupby("bin"):
        a=float(p["sample_weight"].sum())/(float(p["sample_weight"].sum())+strength);bm[int(b)]=a*_wm(p["residual"],p["sample_weight"])+(1-a)*g
    for (stage,b),p in x.groupby(["lifecycle_stage","bin"],dropna=False):
        parent=bm.get(int(b),g);a=float(p["sample_weight"].sum())/(float(p["sample_weight"].sum())+strength)
        sb[("<NA>" if pd.isna(stage) else str(stage),int(b))]=a*_wm(p["residual"],p["sample_weight"])+(1-a)*parent
    return {"edges":edges.tolist(),"global_median":g,"bin_medians":bm,"stage_bin_medians":sb,"oof_rows":len(x),"strength":strength}

def _persist(exp_id,name,scope,dimension,c,pc,ec,pdly,edly,details):
    pcm=_metric(c,"actual_cost_overrun_percentage",pc); ecm=_metric(c,"actual_cost_overrun_percentage",ec)
    pdm=_metric(c,"actual_delay_days",pdly); edm=_metric(c,"actual_delay_days",edly)
    cg,dg=_gain(pcm,ecm),_gain(pdm,edm)
    verdict=("PROMOTION CANDIDATE" if (cg>0 if scope=="cost" else dg>0 if scope=="delay" else cg>=0 and dg>=0 and (cg>0 or dg>0)) else "DO NOT PROMOTE")
    s=c.copy();s["production_cost_prediction"]=pc;s["experiment_cost_prediction"]=ec;s["production_delay_prediction"]=pdly;s["experiment_delay_prediction"]=edly
    cb=paired_project_mae_comparison(s,actual="actual_cost_overrun_percentage",baseline_prediction="production_cost_prediction",candidate_prediction="experiment_cost_prediction",bootstrap_samples=BOOTSTRAP,seed=51000+int(exp_id.split("_")[-1]))
    db=paired_project_mae_comparison(s,actual="actual_delay_days",baseline_prediction="production_delay_prediction",candidate_prediction="experiment_delay_prediction",bootstrap_samples=BOOTSTRAP,seed=61000+int(exp_id.split("_")[-1]))
    extras=[x for x in ["completion_year","lifecycle_stage","sector","implementing_agency","state","project_size_category","approved_cost_cr","cost_escalation_percentage","schedule_slippage_days","duration_ratio","experiment_route"] if x in s]
    ledger=build_prediction_ledger(s,experiment_id=exp_id,window="2001_2021",production_cost_prediction=pc,experiment_cost_prediction=ec,production_delay_prediction=pdly,experiment_delay_prediction=edly,extra_columns=extras)
    assert_prediction_ledger_matches_cohort(ledger,c); run=f"{exp_id}-{uuid.uuid4().hex[:10]}"
    saved=write_experiment_prediction_ledger(ledger,experiment_id=exp_id,window="2001_2021",run_id=run,extra_manifest={"primary_scope":scope,"changed_dimension":dimension,"execution_verdict":"EXECUTION VALID","scientific_verdict":verdict,"decision_window_only":"2001-2021 -> 2022-2025"})
    d=Path(saved["ledger_path"]).parent;(d/"experiment_evidence.json").write_text(json.dumps(_json_safe({"details":details,"cost_bootstrap":cb,"delay_bootstrap":db}),indent=2,allow_nan=False)+"\n")
    overall={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":round(cg,6),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":round(dg,6),"comparison_test_projects":int(c["canonical_project_id"].nunique()),"comparison_test_snapshots":len(c),"paired_project_bootstrap_cost":cb,"paired_project_bootstrap_delay":db,"execution_verdict":"EXECUTION VALID","scientific_verdict":verdict}
    return {"experiment":{"experiment_id":exp_id,"experiment_name":name,"scope":scope,"changed_dimension":dimension,"run_id":run,"promotion_allowed":False,"execution_verdict":"EXECUTION VALID","scientific_verdict":verdict,"ledger_path":str(saved["ledger_path"]),"ledger_manifest_path":str(saved["manifest_path"]),"details":_json_safe(details)},"overall_comparison":overall}

def fit_features(*,data,production_bundle,training_start,training_end,test_end,exp_id,name,scope,dimension,engineer,cost_new=(),delay_new=(),details=None,**_):
    f=engineer(_prepare(data)); train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test)
    cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c),float);pdly=np.maximum(0,np.asarray(dm.predict(c),float));ec=pc.copy();edly=pdly.copy();info=dict(details or {})
    if scope in ("cost","cost+delay"):
        base=list(cm.features); feats=list(dict.fromkeys(base+list(cost_new))); fam=_family(cm);cal=_fit_residual_calibration(_cost_oof(train,feats,fam))
        model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],train,feats,"actual_cost_overrun_percentage");raw=np.asarray(model.predict(c[feats]),float);ec=raw+_corrections(c,raw,cal);info["cost_features_added"]=list(cost_new)
    if scope in ("delay","cost+delay"):
        base=list(dm.features);feats=list(dict.fromkeys(base+list(delay_new)));td=_remaining_frame(train);weights=dict(dm.weights);cal,_=_delay_aft_calibration_oof(td,feats,weights);models=_fit_aft_family_models(td,feats)
        elig=AFTResidualDelayModel._aft_eligible(c).to_numpy(bool);pos=np.flatnonzero(elig)
        if len(pos):
            sub=c.iloc[pos];r=_aft_remaining_prediction(models,weights,sub,feats);raw=_delay_from_remaining(sub,r);edly[pos]=np.maximum(0,raw+_corrections(sub,raw,cal))
        info["delay_features_added"]=list(delay_new);info["aft_eligible_snapshots"]=int(elig.sum())
    return _persist(exp_id,name,scope,dimension,c,pc,ec,pdly,edly,info)

def fit_cost_calibration(*,data,production_bundle,training_start,training_end,test_end,exp_id,name,strength=40.0,**_):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);cm=production_bundle["cost"];fam=_family(cm);base=list(cm.features);cal=shrunk_calibration(_cost_oof(train,base,fam),strength)
    raw=np.asarray(cm.model.predict(c.reindex(columns=base)),float);ec=raw+_corrections(c,raw,cal);pc=np.asarray(cm.predict(c),float);pdly=np.maximum(0,np.asarray(production_bundle["delay"].predict(c),float))
    return _persist(exp_id,name,"cost","fold_stable_shrunk_cost_calibration",c,pc,ec,pdly,pdly.copy(),{"strength":strength})

def fit_delay_calibration(*,data,production_bundle,training_start,training_end,test_end,exp_id,name,strength=40.0,**_):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);dm=production_bundle["delay"];base=list(dm.features);td=_remaining_frame(train)
    chunks=[]
    for fit,val,_year in _rolling_folds(td):
        models=_fit_aft_family_models(fit,base);r=_aft_remaining_prediction(models,dm.weights,val,base);p=_delay_from_remaining(val,r)
        x=val[["actual_delay_days","sample_weight","canonical_project_id","lifecycle_stage"]].copy();x["prediction"]=p;x["residual"]=pd.to_numeric(x["actual_delay_days"],errors="coerce")-p;chunks.append(x)
    cal=shrunk_calibration(pd.concat(chunks,ignore_index=True),strength);pdly=np.maximum(0,np.asarray(dm.predict(c),float));edly=pdly.copy();elig=AFTResidualDelayModel._aft_eligible(c).to_numpy(bool);pos=np.flatnonzero(elig)
    if len(pos):
        sub=c.iloc[pos];r=_aft_remaining_prediction(dm.aft_models,dm.weights,sub,base);raw=_delay_from_remaining(sub,r);edly[pos]=np.maximum(0,raw+_corrections(sub,raw,cal))
    pc=np.asarray(production_bundle["cost"].predict(c),float);return _persist(exp_id,name,"delay","fold_stable_shrunk_aft_calibration",c,pc,pc.copy(),pdly,edly,{"strength":strength,"aft_eligible_snapshots":int(elig.sum())})

def fit_router(*,data,production_bundle,training_start,training_end,test_end,exp_id,name,engineer,score,**_):
    f=engineer(_prepare(data));train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);dm=production_bundle["delay"];pdly=np.maximum(0,np.asarray(dm.predict(c),float));edly=pdly.copy()
    vals=pd.to_numeric(train[score],errors="coerce").dropna();thr=float(vals.quantile(.25));elig=AFTResidualDelayModel._aft_eligible(c).to_numpy(bool);scores=pd.to_numeric(c[score],errors="coerce").fillna(-np.inf).to_numpy(float);route=elig&(scores<thr)
    if route.any():
        pos=np.flatnonzero(route);sub=c.iloc[pos];edly[pos]=np.maximum(0,np.asarray(dm.fallback_model.predict(sub.reindex(columns=dm.features)),float))
    c["experiment_route"]=np.where(route,"reliability_fallback",np.where(elig,"production_aft","production_fallback"));pc=np.asarray(production_bundle["cost"].predict(c),float)
    return _persist(exp_id,name,"delay","planned_date_reliability_evidence_router",c,pc,pc.copy(),pdly,edly,{"training_covariate_threshold_q25":thr,"routed_snapshots":int(route.sum()),"outcome_tuned_threshold":False})

def normalize_taxonomy(f):
    out=f.copy()
    for col in ("sector","implementing_agency"):
        s=out.get(col,pd.Series("<NA>",index=out.index)).astype("string").fillna("<NA>").str.lower().str.replace(r"[^a-z0-9]+"," ",regex=True).str.strip()
        out["_norm_"+col]=s.replace("","<NA>")
    return out

def _prior_map(projects,target,strength=20.0):
    g=float(projects[target].mean()); maps={}
    for keys in [("_norm_implementing_agency","_norm_sector"),("_norm_sector",),("_norm_implementing_agency",)]:
        t=projects.groupby(list(keys),dropna=False)[target].agg(["mean","count"]).reset_index();t["value"]=(t["count"]*t["mean"]+strength*g)/(t["count"]+strength);maps[keys]=t
    return g,maps

def _apply_prior(rows,g,maps):
    val=np.full(len(rows),g,float);sup=np.zeros(len(rows),float);un=np.ones(len(rows),bool)
    for keys in [("_norm_implementing_agency","_norm_sector"),("_norm_sector",),("_norm_implementing_agency",)]:
        m=rows[list(keys)].merge(maps[keys],on=list(keys),how="left",sort=False);found=m["value"].notna().to_numpy()&un;val[found]=m.loc[found,"value"].to_numpy(float);sup[found]=m.loc[found,"count"].to_numpy(float);un[found]=False
    return val,sup

def fit_priors(*,data,production_bundle,training_start,training_end,test_end,exp_id,name,**_):
    f=normalize_taxonomy(_prepare(data));train,test=temporal_project_split(f,training_start,training_end,test_end)
    pp=train.groupby("canonical_project_id",as_index=False).agg(completion_year=("completion_year","first"),actual_cost_overrun_percentage=("actual_cost_overrun_percentage","first"),actual_delay_days=("actual_delay_days","first"),_norm_implementing_agency=("_norm_implementing_agency","first"),_norm_sector=("_norm_sector","first"));pp["completion_year"]=pd.to_numeric(pp["completion_year"],errors="coerce")
    for col in ("exp58_cost_hier_prior","exp58_delay_hier_prior"):train[col]=0.0
    train["exp58_group_support"]=0.0
    for year in sorted(pp["completion_year"].dropna().astype(int).unique()):
        prior=pp[pp["completion_year"]<year];mask=pd.to_numeric(train["completion_year"],errors="coerce").eq(year)
        if prior.empty:continue
        for target,col in [("actual_cost_overrun_percentage","exp58_cost_hier_prior"),("actual_delay_days","exp58_delay_hier_prior")]:
            g,m=_prior_map(prior,target);v,s=_apply_prior(train.loc[mask],g,m);train.loc[mask,col]=v;train.loc[mask,"exp58_group_support"]=np.maximum(train.loc[mask,"exp58_group_support"].to_numpy(float),s)
    test["exp58_group_support"]=0.0
    for target,col in [("actual_cost_overrun_percentage","exp58_cost_hier_prior"),("actual_delay_days","exp58_delay_hier_prior")]:
        g,m=_prior_map(pp,target);v,s=_apply_prior(test,g,m);test[col]=v;test["exp58_group_support"]=np.maximum(test["exp58_group_support"].to_numpy(float),s)
    combined=pd.concat([train,test]).sort_index()
    return fit_features(data=combined,production_bundle=production_bundle,training_start=training_start,training_end=training_end,test_end=test_end,exp_id=exp_id,name=name,scope="cost+delay",dimension="normalized_taxonomy_temporal_hierarchical_priors",engineer=lambda _x:combined.copy(),cost_new=["exp58_cost_hier_prior","exp58_group_support"],delay_new=["exp58_delay_hier_prior","exp58_group_support"],details={"crossfit":"strictly earlier completion years"})

def fit_quality(*,data,production_bundle,training_start,training_end,test_end,exp_id,name,engineer,features,**_):
    f=engineer(_prepare(data));train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);cm,dm=production_bundle["cost"],production_bundle["delay"]
    pc=np.asarray(cm.predict(c),float);pdly=np.maximum(0,np.asarray(dm.predict(c),float))
    base=list(cm.features);allf=list(dict.fromkeys(base+list(features)));fam=_family(cm);cal=_fit_residual_calibration(_cost_oof(train,allf,fam));model=_fit_pipeline(_regressors(PRODUCTION_COST_SEED)[fam],train,allf,"actual_cost_overrun_percentage");raw=np.asarray(model.predict(c[allf]),float);ec=raw+_corrections(c,raw,cal)
    edly=pdly.copy();pt=pd.to_datetime(train.get("planned_completion_date"),errors="coerce");ft=train.loc[pt.isna()].copy();mask=pd.to_datetime(c.get("planned_completion_date"),errors="coerce").isna().to_numpy(bool);used=False
    if ft["canonical_project_id"].nunique()>=20 and len(ft)>=100 and mask.any():
        ft=assign_project_balanced_weights(ft);df=list(dict.fromkeys(list(dm.features)+list(features)));special=_fit_pipeline(_regressors(26204)["extra_trees"],ft,df,"actual_delay_days");pos=np.flatnonzero(mask);edly[pos]=np.maximum(0,np.asarray(special.predict(c.iloc[pos][df]),float));used=True
    c["experiment_route"]=np.where(mask,"missing_planned_specialist" if used else "production_fallback","production_aft_or_fallback")
    return _persist(exp_id,name,"cost+delay","data_quality_features_plus_missing_planned_fallback_specialist",c,pc,ec,pdly,edly,{"features":list(features),"fallback_training_projects":int(ft["canonical_project_id"].nunique()),"fallback_holdout_snapshots":int(mask.sum()),"specialist_used":used})

def run_cli(module):
    p=argparse.ArgumentParser();p.add_argument("--start",type=int,default=2001);p.add_argument("--end",type=int,default=2021);p.add_argument("--test-end",type=int,default=2025);p.add_argument("--output",required=True);a=p.parse_args()
    if (a.start,a.end,a.test_end)!=(2001,2021,2025):raise ValueError("Exp51-60 batch supports only 2001-2021 -> 2022-2025")
    before=_hash_prod();data,identity=build_training_dataset()
    with tempfile.TemporaryDirectory(prefix=module.EXPERIMENT_ID+"-") as td:
        root=Path(td)/"production";receipt=train_window_with_promoted_cost_and_delay(2001,2021,2025,data=data,identity=identity,artifact_root=root);t=root/"2001_2021";bundle={"cost":joblib.load(t/"cost_model.pkl"),"delay":joblib.load(t/"delay_model.pkl"),"metadata":json.loads((t/"metadata.json").read_text())}
        fitted=module.fit_experiment(data=data,training_start=2001,training_end=2021,test_end=2025,production_bundle=bundle,production_receipt=receipt)
    if before!=_hash_prod():raise AssertionError("production artifacts changed")
    o=fitted["overall_comparison"];obs=(o["comparison_test_projects"],o["comparison_test_snapshots"],float(o["production_cost_mae"]),float(o["production_delay_mae"]))
    if obs[0]!=721 or obs[1]!=11200 or abs(obs[2]-26.287)>.005 or abs(obs[3]-431.618)>.005:print(module.MARKER+"_EXECUTION_VERDICT=EXECUTION INVALID");raise RuntimeError(obs)
    payload={"window":"2001_2021","test_end":2025,"production":receipt,"experiment":fitted["experiment"],"overall_comparison":o,"production_artifacts_untouched":True};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(_json_safe(payload),indent=2,allow_nan=False)+"\n")
    q=module.MARKER+"_2001_2021";print(q+"_PRODUCTION_COST_MAE="+str(o["production_cost_mae"]));print(q+"_EXPERIMENT_COST_MAE="+str(o["experiment_cost_mae"]));print(q+"_COST_IMPROVEMENT_PERCENT="+str(o["cost_improvement_percentage"]));print(q+"_PRODUCTION_DELAY_MAE="+str(o["production_delay_mae"]));print(q+"_EXPERIMENT_DELAY_MAE="+str(o["experiment_delay_mae"]));print(q+"_DELAY_IMPROVEMENT_PERCENT="+str(o["delay_improvement_percentage"]));print(module.MARKER+"_EXECUTION_VERDICT=EXECUTION VALID");print(module.MARKER+"_SCIENTIFIC_VERDICT="+o["scientific_verdict"])
