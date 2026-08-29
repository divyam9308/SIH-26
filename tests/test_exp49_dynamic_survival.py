from pathlib import Path
import numpy as np
import pandas as pd
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.dynamic_survival_exp49 import FORBIDDEN_INPUTS, _enrich_archive, build_survival_risk_set
from backend.app.ml.experiments.framework import experiment_run_directory
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights


def _history():
    rows = []
    for project, completion in [("DONE", "2020-10-01"), ("ACTIVE", pd.NaT), ("LATER", "2023-06-01")]:
        for i in range(3):
            rows.append({"canonical_project_id": project, "snapshot_date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=3*i), "planned_start_date": "2019-01-01", "approval_date": "2018-12-01", "completion_date": completion, "approved_cost_cr": 100.0, "revised_cost_cr": 100 + 5*i, "cumulative_expenditure_cr": 10 + 10*i, "cost_escalation_percentage": 5*i, "schedule_slippage_days": 30*i, "duration_ratio": .2 + .1*i})
    return pd.DataFrame(rows)


def test_censoring_does_not_fabricate_later_or_unknown_completion():
    risk = build_survival_risk_set(_history(), cutoff="2021-12-31")
    events = risk.groupby("canonical_project_id").exp49_event.sum().to_dict()
    assert events["DONE"] == 1
    assert events["ACTIVE"] == 0
    assert events["LATER"] == 0
    assert (risk.groupby("canonical_project_id").exp49_event.sum() <= 1).all()


def test_intervals_are_positive_ordered_project_balanced_and_excludable():
    risk = build_survival_risk_set(_history(), cutoff="2021-12-31", excluded_project_ids={"LATER"})
    assert "LATER" not in set(risk.canonical_project_id)
    assert (risk.exp49_stop_days > risk.exp49_start_days).all()
    assert np.allclose(risk.groupby("canonical_project_id").sample_weight.sum(), 1.0)


def test_future_append_does_not_change_earlier_covariates():
    history = _history(); before = _enrich_archive(history)
    future = pd.concat([history, pd.DataFrame([{**history.iloc[-1].to_dict(), "canonical_project_id": "ACTIVE", "snapshot_date": "2025-01-01", "revised_cost_cr": 900.0, "completion_date": "2026-01-01"}])], ignore_index=True)
    after = _enrich_archive(future)
    columns = [column for column in before.columns if column not in FORBIDDEN_INPUTS | {"completion_date", "completion_year", "actual_risk"}]
    left = before[before.canonical_project_id.eq("ACTIVE")][columns].reset_index(drop=True)
    right = after[(after.canonical_project_id.eq("ACTIVE")) & after.snapshot_date.lt("2025-01-01")][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_full_delay_ledger_route_and_weights():
    rows = assign_project_balanced_weights(pd.DataFrame({"canonical_project_id": ["A", "A", "B"], "snapshot_date": pd.to_datetime(["2022-01-01", "2022-04-01", "2022-01-01"]), "actual_delay_days": [10., 20., 30.], "experiment_route": ["exp49_survival_aft_blend", "exp49_survival_aft_blend", "exp34_fallback"]}))
    ledger = build_prediction_ledger(rows, experiment_id="exp_49", window="2001_2021", production_delay_prediction=[11, 19, 35], experiment_delay_prediction=[10, 18, 35], extra_columns=["experiment_route"])
    assert_prediction_ledger_matches_cohort(ledger, rows)
    assert set(ledger.experiment_route) == {"exp49_survival_aft_blend", "exp34_fallback"}


def test_adapter_forbidden_inputs_and_production_safety():
    assert get_experiment_adapter("exp_49").sequence == 49
    assert "actual_delay_days" in FORBIDDEN_INPUTS and "actual_completion_date" in FORBIDDEN_INPUTS
    destination = experiment_run_directory("exp_49", "2001_2021", "test-run"); assert "experiments/exp_49/2001_2021/test-run" in destination.as_posix()
    source = Path("backend/app/ml/experiments/dynamic_survival_exp49.py").read_text(); assert "joblib.dump" not in source; assert "candidate_cost = production_cost.copy()" in source
    assert "holdout_ids" in source and "excluded_project_ids=holdout_ids" in source
