from pathlib import Path
import numpy as np
import pandas as pd
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.forward_schedule_revision_exp50 import AUXILIARY_INPUT_FEATURES,EXP50_FEATURES,FORBIDDEN_INPUTS,MIN_SCHEDULE_REVISION_DAYS,build_forward_schedule_revision_dataset
from backend.app.ml.experiments.framework import experiment_run_directory
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort,build_prediction_ledger
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights

def _history():
    rows=[]
    for month,move in [(0,0),(2,5),(4,45),(8,45),(13,100)]:
        planned=pd.Timestamp("2021-01-01");rows.append({"canonical_project_id":"A","snapshot_date":pd.Timestamp("2020-01-01")+pd.DateOffset(months=month),"planned_completion_date":planned,"revised_completion_date":planned+pd.Timedelta(days=move),"approved_cost_cr":100.,"revised_cost_cr":100+month,"cumulative_expenditure_cr":5+3*month,"schedule_slippage_days":move,"schedule_slippage_ratio":move/730,"duration_ratio":(month+1)/24,"cost_escalation_percentage":month,"expenditure_ratio":(5+3*month)/100})
    return pd.DataFrame(rows)

def test_future_append_cannot_change_earlier_auxiliary_inputs():
    history=_history();before=build_forward_schedule_revision_dataset(history);future=pd.concat([history,pd.DataFrame([{**history.iloc[-1].to_dict(),"snapshot_date":"2025-01-01","revised_completion_date":"2035-01-01"}])],ignore_index=True);after=build_forward_schedule_revision_dataset(future);columns=["canonical_project_id","snapshot_date",*AUXILIARY_INPUT_FEATURES];pd.testing.assert_frame_equal(before[columns],after.iloc[:len(before)][columns],check_dtype=False)
def test_labels_ignore_small_noise_and_censor_unknown_tail():
    frame=build_forward_schedule_revision_dataset(_history());assert MIN_SCHEDULE_REVISION_DAYS==14.;assert frame.loc[0,"schedule_revision_within_6m"]==1.;assert frame.loc[0,"next_schedule_revision_days"]==40.;assert pd.isna(frame.iloc[-1]["schedule_revision_within_3m"]);assert frame.iloc[-1].auxiliary_next_revision_observed==0
def test_duplicates_deterministic_and_forbidden_inputs_absent():
    history=pd.concat([_history(),_history().iloc[[2]].assign(revised_completion_date="2021-03-01")],ignore_index=True);first=build_forward_schedule_revision_dataset(history);second=build_forward_schedule_revision_dataset(history.sample(frac=1,random_state=50));pd.testing.assert_frame_equal(first,second);assert not first.duplicated(["canonical_project_id","snapshot_date"]).any();assert not(FORBIDDEN_INPUTS&set(AUXILIARY_INPUT_FEATURES));assert len(EXP50_FEATURES)==7
def test_delay_ledger_routes_and_project_weights():
    rows=assign_project_balanced_weights(pd.DataFrame({"canonical_project_id":["A","A","B"],"snapshot_date":pd.to_datetime(["2022-01-01","2022-04-01","2022-01-01"]),"actual_delay_days":[10.,20.,30.],"experiment_route":["exp50_aft","exp50_aft","exp34_fallback"]}));assert np.allclose(rows.groupby("canonical_project_id").sample_weight.sum(),1);ledger=build_prediction_ledger(rows,experiment_id="exp_50",window="2001_2021",production_delay_prediction=[11,19,35],experiment_delay_prediction=[10,18,35],extra_columns=["experiment_route"]);assert_prediction_ledger_matches_cohort(ledger,rows);assert set(ledger.experiment_route)=={"exp50_aft","exp34_fallback"}
def test_adapter_no_exp46_dependency_and_production_safety():
    assert get_experiment_adapter("exp_50").sequence==50;destination=experiment_run_directory("exp_50","2001_2021","test-run");assert "experiments/exp_50/2001_2021/test-run" in destination.as_posix();source=Path("backend/app/ml/experiments/forward_schedule_revision_exp50.py").read_text();assert "exp46_" not in source.lower();assert "joblib.dump" not in source;assert "candidate_cost=production_cost.copy()" in source;assert "holdout_ids" in source
