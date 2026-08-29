"""Experiment 50: cross-fitted forward schedule-revision representation."""
from __future__ import annotations
import json, math, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import _aft_remaining_prediction, _corrections, _delay_aft_calibration_oof, _delay_from_remaining, _fit_aft_family_models, _public_calibration, _remaining_frame
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger, write_experiment_prediction_ledger
from backend.app.ml.experiments.trajectory_exp12 import EXP12_FEATURES, engineer_history
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _json_safe, _preprocessor, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows, enrich_supervised_for_production
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel, CALIBRATION_GATE_FEATURE, _select_aft_calibration_projects

EXPERIMENT_ID="exp_50"; EXPERIMENT_SEQUENCE=50
EXPERIMENT_NAME="Full-archive forward schedule-revision representation"; EXPERIMENT_SCOPE="delay"
HYPOTHESIS="Cross-fitted predictions of recurrent intermediate schedule revisions add leading remaining-time information beyond current Exp32/Exp34 path features."
CHANGED_DIMENSION="cross_fitted_intermediate_supervision"
MIN_SCHEDULE_REVISION_DAYS=14.0; AUXILIARY_FOLDS=3; AUXILIARY_SEED=50050
HORIZON_DAYS={3:92,6:183,12:366}
SOURCE_COLUMNS={"canonical_project_id","snapshot_date","planned_completion_date","revised_completion_date","approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","schedule_slippage_days","duration_ratio"}
FORBIDDEN_INPUTS={"completion_date","actual_completion_date","actual_delay_days","actual_cost_overrun_percentage","reported_completion_expenditure_cr"}
AUXILIARY_INPUT_FEATURES=[
    "approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","cost_escalation_percentage","expenditure_ratio","schedule_slippage_days","schedule_slippage_ratio","duration_ratio",
    "exp50_history_observations","exp50_days_since_previous_report","exp50_prior_schedule_revision_count","exp50_months_since_prior_schedule_revision","exp50_previous_schedule_revision_days","exp50_prior_schedule_revision_abs_mean","exp50_prior_extension_share",
    *EXP12_FEATURES,
    "exp34_observations_seen","exp34_months_observed","exp34_cost_revision_count","exp34_schedule_revision_count","exp34_cumulative_abs_cost_revision_pct","exp34_max_cost_escalation","exp34_cost_recovery_from_peak","exp34_max_schedule_slippage","exp34_delay_recovery_from_peak","exp34_slippage_positive_share","exp34_cost_overrun_positive_share","exp34_cost_worsening_share","exp34_delay_worsening_share","exp34_months_since_first_cost_revision","exp34_months_since_first_schedule_revision",
]
EXP50_FEATURES=["exp50_revision_probability_3m","exp50_revision_probability_6m","exp50_revision_probability_12m","exp50_days_to_next_revision_prediction","exp50_next_revision_magnitude_days_prediction","exp50_extension_probability","exp50_revision_count_next_12m_prediction"]
AUXILIARY_LABELS=["schedule_revision_within_3m","schedule_revision_within_6m","schedule_revision_within_12m","days_to_next_schedule_revision","next_schedule_revision_days","next_schedule_revision_extension","schedule_revision_count_next_12m"]


class _ConstantClassifier:
    def __init__(self,p): self.p=float(np.clip(p,0,1))
    def predict_proba(self,frame):
        p=np.full(len(frame),self.p); return np.column_stack([1-p,p])
class _ConstantRegressor:
    def __init__(self,value): self.value=float(value)
    def predict(self,frame): return np.full(len(frame),self.value)


def _canonical_history(history):
    missing=sorted(SOURCE_COLUMNS.difference(history.columns))
    if missing: raise ValueError("Exp50 history is missing: "+", ".join(missing))
    frame=history.copy(); frame["canonical_project_id"]=frame.canonical_project_id.astype("string").str.strip(); frame["snapshot_date"]=pd.to_datetime(frame.snapshot_date,errors="coerce")
    for name in ("planned_completion_date","revised_completion_date"): frame[name]=pd.to_datetime(frame[name],errors="coerce")
    for name in ("approved_cost_cr","revised_cost_cr","cumulative_expenditure_cr","schedule_slippage_days","duration_ratio"): frame[name]=pd.to_numeric(frame[name],errors="coerce")
    frame=frame.dropna(subset=["canonical_project_id","snapshot_date"])
    safe=frame.reindex(columns=sorted(SOURCE_COLUMNS)).astype("string").fillna("<NA>"); frame["_exp50_tie"]=pd.util.hash_pandas_object(safe,index=False).to_numpy(np.uint64)
    return frame.sort_values(["canonical_project_id","snapshot_date","_exp50_tie"],kind="mergesort").drop_duplicates(["canonical_project_id","snapshot_date"],keep="last").drop(columns="_exp50_tie").reset_index(drop=True)


def _effective_dates(frame): return pd.to_datetime(frame.revised_completion_date.where(frame.revised_completion_date.notna(),frame.planned_completion_date),errors="coerce")


def _add_prefix_features(frame):
    result=enrich_path_dependence(engineer_history(frame),history=frame)
    for name in ("exp50_history_observations","exp50_days_since_previous_report","exp50_prior_schedule_revision_count","exp50_months_since_prior_schedule_revision","exp50_previous_schedule_revision_days","exp50_prior_schedule_revision_abs_mean","exp50_prior_extension_share"): result[name]=0.0
    result["exp50_days_since_previous_report"]=-1.0; result["exp50_months_since_prior_schedule_revision"]=-1.0
    effective=_effective_dates(result)
    for _,group in result.groupby("canonical_project_id",sort=False):
        events=[]; event_positions=[]; indices=group.index.to_numpy(); dates=group.snapshot_date.tolist()
        for pos,index in enumerate(indices):
            result.at[index,"exp50_history_observations"]=pos+1
            if pos: result.at[index,"exp50_days_since_previous_report"]=(dates[pos]-dates[pos-1]).days
            if pos and pd.notna(effective.loc[index]) and pd.notna(effective.loc[indices[pos-1]]):
                move=float((effective.loc[index]-effective.loc[indices[pos-1]]).days)
                if abs(move)>=MIN_SCHEDULE_REVISION_DAYS: events.append(move); event_positions.append(pos)
            result.at[index,"exp50_prior_schedule_revision_count"]=len(events)
            if events:
                result.at[index,"exp50_months_since_prior_schedule_revision"]=(dates[pos]-dates[event_positions[-1]]).days/30.4375
                result.at[index,"exp50_previous_schedule_revision_days"]=events[-1]
                result.at[index,"exp50_prior_schedule_revision_abs_mean"]=float(np.mean(np.abs(events)))
                result.at[index,"exp50_prior_extension_share"]=float(np.mean(np.asarray(events)>0))
    return result


def build_forward_schedule_revision_dataset(history,*,cutoff=None):
    frame=_canonical_history(history)
    if cutoff is not None: frame=frame[frame.snapshot_date.le(pd.Timestamp(cutoff))].copy()
    frame=_add_prefix_features(frame)
    for label in AUXILIARY_LABELS: frame[label]=np.nan
    frame["auxiliary_followup_days"]=0.0; frame["auxiliary_next_revision_observed"]=0.0
    effective=_effective_dates(frame)
    for _,group in frame.groupby("canonical_project_id",sort=False):
        indices=group.index.to_numpy(); dates=group.snapshot_date.to_numpy(dtype="datetime64[ns]"); positions=[]; moves=[]
        for pos in range(1,len(group)):
            current,previous=effective.loc[indices[pos]],effective.loc[indices[pos-1]]
            if pd.notna(current) and pd.notna(previous):
                move=float((current-previous).days)
                if abs(move)>=MIN_SCHEDULE_REVISION_DAYS: positions.append(pos); moves.append(move)
        event_array=np.asarray(positions,dtype=int)
        for pos,index in enumerate(indices):
            followup=float((dates[-1]-dates[pos])/np.timedelta64(1,"D")); frame.at[index,"auxiliary_followup_days"]=max(0,followup)
            pointer=int(np.searchsorted(event_array,pos+1)) if len(event_array) else 0; has=pointer<len(positions)
            days=float((dates[positions[pointer]]-dates[pos])/np.timedelta64(1,"D")) if has else math.nan
            for months,horizon in HORIZON_DAYS.items():
                label=f"schedule_revision_within_{months}m"
                if has and days<=horizon: frame.at[index,label]=1.0
                elif followup>=horizon: frame.at[index,label]=0.0
            future12=[move for event_pos,move in zip(positions,moves) if event_pos>pos and float((dates[event_pos]-dates[pos])/np.timedelta64(1,"D"))<=366]
            if followup>=366: frame.at[index,"schedule_revision_count_next_12m"]=len(future12)
            if has:
                move=moves[pointer]; frame.at[index,"auxiliary_next_revision_observed"]=1.0; frame.at[index,"days_to_next_schedule_revision"]=days; frame.at[index,"next_schedule_revision_days"]=move; frame.at[index,"next_schedule_revision_extension"]=float(move>0)
    return frame


def _fit_classifier(rows,label,seed):
    available=rows.dropna(subset=[label]).copy()
    if available.empty:return _ConstantClassifier(0)
    available=assign_project_balanced_weights(available); p=float(available[label].mean())
    if available[label].nunique()<2:return _ConstantClassifier(p)
    model=ExtraTreesClassifier(n_estimators=120,min_samples_leaf=5,max_features=.8,class_weight="balanced_subsample",random_state=seed,n_jobs=2); pipe=Pipeline([("preprocess",_preprocessor(available,AUXILIARY_INPUT_FEATURES)),("model",model)]); pipe.fit(available[AUXILIARY_INPUT_FEATURES],available[label].astype(int),model__sample_weight=available.sample_weight.to_numpy(float)); return pipe
def _fit_regressor(rows,label,seed):
    available=rows.dropna(subset=[label]).copy()
    if available.empty:return _ConstantRegressor(0)
    available=assign_project_balanced_weights(available)
    if available[label].nunique()<2:return _ConstantRegressor(available[label].median())
    model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=5,max_features=.8,random_state=seed,n_jobs=2); pipe=Pipeline([("preprocess",_preprocessor(available,AUXILIARY_INPUT_FEATURES)),("model",model)]); pipe.fit(available[AUXILIARY_INPUT_FEATURES],available[label],model__sample_weight=available.sample_weight.to_numpy(float)); return pipe
def fit_auxiliary_models(rows): return {"e3":_fit_classifier(rows,"schedule_revision_within_3m",AUXILIARY_SEED+3),"e6":_fit_classifier(rows,"schedule_revision_within_6m",AUXILIARY_SEED+6),"e12":_fit_classifier(rows,"schedule_revision_within_12m",AUXILIARY_SEED+12),"days":_fit_regressor(rows,"days_to_next_schedule_revision",AUXILIARY_SEED+20),"magnitude":_fit_regressor(rows,"next_schedule_revision_days",AUXILIARY_SEED+21),"extension":_fit_classifier(rows,"next_schedule_revision_extension",AUXILIARY_SEED+22),"count":_fit_regressor(rows,"schedule_revision_count_next_12m",AUXILIARY_SEED+23)}
def _prob(model,rows):return np.asarray(model.predict_proba(rows[AUXILIARY_INPUT_FEATURES]))[:,1]
def predict_auxiliary(models,rows):
    result=rows[["canonical_project_id","snapshot_date"]].copy(); result["exp50_revision_probability_3m"]=_prob(models["e3"],rows); result["exp50_revision_probability_6m"]=_prob(models["e6"],rows); result["exp50_revision_probability_12m"]=_prob(models["e12"],rows); result["exp50_days_to_next_revision_prediction"]=np.maximum(0,models["days"].predict(rows[AUXILIARY_INPUT_FEATURES])); result["exp50_next_revision_magnitude_days_prediction"]=models["magnitude"].predict(rows[AUXILIARY_INPUT_FEATURES]); result["exp50_extension_probability"]=_prob(models["extension"],rows); result["exp50_revision_count_next_12m_prediction"]=np.maximum(0,models["count"].predict(rows[AUXILIARY_INPUT_FEATURES])); return result


def _rows_for_keys(source,target):
    keys=target[["canonical_project_id","snapshot_date"]].copy(); keys["canonical_project_id"]=keys.canonical_project_id.astype("string"); keys["snapshot_date"]=pd.to_datetime(keys.snapshot_date)
    columns=["canonical_project_id","snapshot_date",*AUXILIARY_INPUT_FEATURES,*AUXILIARY_LABELS]; available=source.reindex(columns=columns).drop_duplicates(["canonical_project_id","snapshot_date"],keep="last"); result=keys.merge(available,on=["canonical_project_id","snapshot_date"],how="left",validate="many_to_one")
    if len(result)!=len(target):raise AssertionError("Exp50 auxiliary lookup changed the cohort")
    return result


def _diagnostics(rows,predictions,pool):
    result={"meaningful_revision_threshold_days":MIN_SCHEDULE_REVISION_DAYS,"training_rows":len(pool),"training_projects":pool.canonical_project_id.nunique(),"observed_revision_events":int(pool.groupby("canonical_project_id")["exp50_prior_schedule_revision_count"].max().sum()),"censored_or_no_next_event_rows":int((pool.auxiliary_next_revision_observed==0).sum())}
    for months,prediction in ((3,"exp50_revision_probability_3m"),(6,"exp50_revision_probability_6m"),(12,"exp50_revision_probability_12m")):
        label=f"schedule_revision_within_{months}m"; mask=rows[label].notna(); y=rows.loc[mask,label].to_numpy(float); p=predictions.loc[mask,prediction].to_numpy(float); item={"available_rows":int(mask.sum()),"prevalence":float(y.mean()) if len(y) else None}
        if len(np.unique(y))>1:item.update({"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),"brier_score":brier_score_loss(y,p)})
        result[f"event_{months}m"]=item
    for label,prediction,name in (("days_to_next_schedule_revision","exp50_days_to_next_revision_prediction","time_to_next_revision"),("next_schedule_revision_days","exp50_next_revision_magnitude_days_prediction","next_revision_magnitude")):
        mask=rows[label].notna(); result[name]={"available_rows":int(mask.sum()),"mae":mean_absolute_error(rows.loc[mask,label],predictions.loc[mask,prediction]) if mask.any() else None}
    return _json_safe(result)


def cross_fitted_auxiliary_features(history,train,holdout,*,training_end):
    training_archive=build_forward_schedule_revision_dataset(history,cutoff=f"{training_end}-12-31"); prediction_archive=build_forward_schedule_revision_dataset(history); holdout_ids=set(holdout.canonical_project_id.astype("string")); pool=training_archive[~training_archive.canonical_project_id.astype("string").isin(holdout_ids)].copy()
    train_rows=_rows_for_keys(training_archive,train); holdout_rows=_rows_for_keys(prediction_archive,holdout); projects=np.asarray(sorted(train.canonical_project_id.astype("string").unique())); rng=np.random.default_rng(AUXILIARY_SEED+training_end); rng.shuffle(projects); folds=[x for x in np.array_split(projects,min(AUXILIARY_FOLDS,len(projects))) if len(x)]; parts=[]
    for fold in folds:
        ids=set(fold.tolist()); models=fit_auxiliary_models(pool[~pool.canonical_project_id.astype("string").isin(ids)]); validation=train_rows[train_rows.canonical_project_id.astype("string").isin(ids)]; predicted=predict_auxiliary(models,validation); predicted["_order"]=validation.index; parts.append(predicted)
    train_features=pd.concat(parts,ignore_index=True).sort_values("_order").drop(columns="_order").reset_index(drop=True); holdout_features=predict_auxiliary(fit_auxiliary_models(pool),holdout_rows).reset_index(drop=True)
    if len(train_features)!=len(train):raise AssertionError("Exp50 OOF features do not match final training rows")
    diagnostics=_diagnostics(train_rows.reset_index(drop=True),train_features,pool); diagnostics.update({"project_grouped_oof_folds":len(folds),"holdout_projects_excluded_from_auxiliary_training":len(holdout_ids),"auxiliary_training_cutoff":f"{training_end}-12-31","identity_rule":"stable canonical history; no fuzzy outcome linkage"}); return train_features,holdout_features,_json_safe(diagnostics)


def _route_metrics(rows,mask):
    part=rows.loc[mask]
    if part.empty:return {"projects":0,"snapshots":0,"production_delay_mae":None,"experiment_delay_mae":None}
    weighted=assign_project_balanced_weights(part); return {"projects":weighted.canonical_project_id.nunique(),"snapshots":len(weighted),"production_delay_mae":_regression_metrics(weighted.actual_delay_days,weighted.production_delay_prediction.to_numpy(),weighted.sample_weight,weighted.canonical_project_id)["MAE"],"experiment_delay_mae":_regression_metrics(weighted.actual_delay_days,weighted.experiment_delay_prediction.to_numpy(),weighted.sample_weight,weighted.canonical_project_id)["MAE"]}


def _write_artifacts(directory,auxiliary,bootstrap,features,comparison):
    payloads={"auxiliary_label_audit.json":{"threshold_days":MIN_SCHEDULE_REVISION_DAYS,"labels":AUXILIARY_LABELS,"future_reports_are_labels_only":True},"auxiliary_model_diagnostics.json":auxiliary,"model_feature_config.json":{"auxiliary_family":"fixed ExtraTrees","oof_folds":AUXILIARY_FOLDS,"final_features":features},"bootstrap_results.json":bootstrap}
    for name,payload in payloads.items():(directory/name).write_text(json.dumps(_json_safe(payload),indent=2,allow_nan=False)+"\n")
    (directory/"experiment_summary.md").write_text(f"# Experiment 50 comparison\n\n{HYPOTHESIS}\n\nProduction Delay MAE: {comparison['production_delay_mae']}\n\nExperiment Delay MAE: {comparison['experiment_delay_mae']}\n\nImprovement: {comparison['delay_improvement_percentage']}%\n\nExecution verdict: EXECUTION VALID\n\nScientific verdict: {comparison['scientific_verdict']}\n"); return {name:str(directory/name) for name in [*payloads,"experiment_summary.md"]}


def fit_experiment(*,data,training_start,training_end,test_end,production_bundle,production_receipt,history=None,**_):
    if history is None:
        if not TRAJECTORIES.exists():raise FileNotFoundError("Exp50 requires full monthly trajectories")
        history=pd.read_csv(TRAJECTORIES,dtype={"canonical_project_id":"string"},low_memory=False)
    base=enrich_path_dependence(enrich_supervised_for_production(data.copy())); base["completion_year"]=pd.to_numeric(base.completion_year,errors="coerce"); base["snapshot_date"]=pd.to_datetime(base.snapshot_date,errors="coerce"); base["canonical_project_id"]=base.canonical_project_id.astype("string"); train,test=temporal_project_split(base,training_start,training_end,test_end)
    train_aux,test_aux,auxiliary=cross_fitted_auxiliary_features(history,train,test,training_end=training_end); train=train.merge(train_aux,on=["canonical_project_id","snapshot_date"],how="left",validate="one_to_one"); test=test.merge(test_aux,on=["canonical_project_id","snapshot_date"],how="left",validate="one_to_one")
    for feature in EXP50_FEATURES:train[feature]=pd.to_numeric(train[feature],errors="coerce").fillna(0);test[feature]=pd.to_numeric(test[feature],errors="coerce").fillna(0)
    production_model=production_bundle["delay"]
    if not isinstance(production_model,AFTResidualDelayModel):raise TypeError("Exp50 requires current production AFTResidualDelayModel")
    base_features=list(production_model.features); features=list(dict.fromkeys(base_features+EXP50_FEATURES)); weights=dict(production_model.weights)
    train_delay=_remaining_frame(train); calibration,oof=_delay_aft_calibration_oof(train_delay,features,weights); models=_fit_aft_family_models(train_delay,features)
    compare=_production_cost_evaluation_rows(test).copy(); gate=_select_aft_calibration_projects(compare); compare[CALIBRATION_GATE_FEATURE]=compare.canonical_project_id.astype("string").isin(gate); compare=assign_project_balanced_weights(compare).reset_index(drop=True)
    production_delay=np.maximum(0,production_model.predict(compare)); candidate_delay=production_delay.copy(); route=production_model._aft_eligible(compare).to_numpy(bool)
    if route.any():
        positions=np.flatnonzero(route); subset=compare.iloc[positions]; remaining=_aft_remaining_prediction(models,weights,subset,features); raw=_delay_from_remaining(subset,remaining); candidate_delay[positions]=np.maximum(0,raw+_corrections(subset,raw,calibration))
    if not np.array_equal(candidate_delay[~route],production_delay[~route]):raise AssertionError("Exp50 changed fallback")
    production_cost=production_bundle["cost"].predict(compare); candidate_cost=production_cost.copy()
    prod=_regression_metrics(compare.actual_delay_days,production_delay,compare.sample_weight,compare.canonical_project_id); exp=_regression_metrics(compare.actual_delay_days,candidate_delay,compare.sample_weight,compare.canonical_project_id); cost=_regression_metrics(compare.actual_cost_overrun_percentage,production_cost,compare.sample_weight,compare.canonical_project_id); absolute=float(prod["MAE"])-float(exp["MAE"]); percentage=absolute/float(prod["MAE"])*100
    scored=compare.copy(); scored["production_delay_prediction"]=production_delay; scored["experiment_delay_prediction"]=candidate_delay; scored["experiment_route"]=np.where(route,"exp50_aft","exp34_fallback"); bootstrap=paired_project_mae_comparison(scored,actual="actual_delay_days",baseline_prediction="production_delay_prediction",candidate_prediction="experiment_delay_prediction",bootstrap_samples=5000,seed=50000+training_end); verdict="PROMOTION CANDIDATE" if percentage>0 and bootstrap["probability_candidate_better"]>=.5 else "DO NOT PROMOTE"; aft=_route_metrics(scored,route); fallback=_route_metrics(scored,~route)
    if fallback["production_delay_mae"]!=fallback["experiment_delay_mae"]:raise AssertionError("Exp50 fallback metrics differ")
    comparison={"production_delay_mae":prod["MAE"],"experiment_delay_mae":exp["MAE"],"absolute_delay_mae_improvement":round(absolute,6),"delay_improvement_percentage":round(percentage,6),"production_cost_mae":cost["MAE"],"experiment_cost_mae":cost["MAE"],"cost_predictions_identical":True,"comparison_test_projects":compare.canonical_project_id.nunique(),"comparison_test_snapshots":len(compare),"aft_projects":scored.loc[route,"canonical_project_id"].nunique(),"fallback_projects":scored.loc[~route,"canonical_project_id"].nunique(),"aft_snapshots":int(route.sum()),"fallback_snapshots":int((~route).sum()),"aft_route":aft,"fallback_route":fallback,"paired_project_bootstrap":bootstrap,"execution_verdict":"EXECUTION VALID","scientific_verdict":verdict}
    run_id=f"exp50-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}";window=f"{training_start}_{training_end}";ledger=build_prediction_ledger(scored,experiment_id=EXPERIMENT_ID,window=window,production_delay_prediction=production_delay,experiment_delay_prediction=candidate_delay,extra_columns=["completion_year","lifecycle_stage","sector","implementing_agency","state","project_size_category","approved_cost_cr","schedule_slippage_days","duration_ratio","exp12_history_12m","exp34_observations_seen","experiment_route"]);assert_prediction_ledger_matches_cohort(ledger,compare);persisted=write_experiment_prediction_ledger(ledger,experiment_id=EXPERIMENT_ID,window=window,run_id=run_id,extra_manifest={"primary_target":"delay","execution_verdict":"EXECUTION VALID","scientific_verdict":verdict,"changed_dimension":CHANGED_DIMENSION,"bootstrap_samples":5000,"cost_unchanged":True,"fallback_unchanged":True,"auxiliary_revision_events":auxiliary["observed_revision_events"]});artifacts=_write_artifacts(Path(persisted["ledger_path"]).parent,auxiliary,bootstrap,features,comparison);lookup={(str(row.canonical_project_id),pd.Timestamp(row.snapshot_date).isoformat()):float(pred) for (_,row),pred in zip(scored.iterrows(),candidate_delay)}
    return {"experiment":{"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"scope":EXPERIMENT_SCOPE,"run_id":run_id,"model_role":"experiment","promotion_allowed":False,"changed_dimension":CHANGED_DIMENSION,"hypothesis":HYPOTHESIS,"new_features":EXP50_FEATURES,"fixed_production_aft_weights":weights,"calibration_method":"production Exp33 rolling-OOF method refit for challenger","rolling_oof":oof,"calibration":_public_calibration(calibration),"auxiliary_diagnostics":auxiliary,"future_holdout_used_for_training_or_selection":False,"execution_verdict":"EXECUTION VALID","scientific_verdict":verdict,"ledger_path":str(persisted["ledger_path"]),"ledger_manifest_path":str(persisted["manifest_path"]),"cohort_fingerprint":persisted["manifest"]["cohort_fingerprint"],"ledger_fingerprint":persisted["manifest"]["ledger_fingerprint"],"audit_artifacts":artifacts},"overall_comparison":comparison,"state":{"lookup":lookup}}


def filter_comparable_rows(frame,state):
    keys=set(state.get("lookup",{}));mask=[(str(row.canonical_project_id),pd.Timestamp(row.snapshot_date).isoformat()) in keys for _,row in frame.iterrows()];return assign_project_balanced_weights(frame.loc[mask].copy())
def predict_project(row,state):
    key=(str(row.canonical_project_id),pd.Timestamp(row.snapshot_date).isoformat())
    if key not in state.get("lookup",{}):raise ValueError("Exp50 row outside frozen cohort")
    return {"delay_days":float(state["lookup"][key])}
