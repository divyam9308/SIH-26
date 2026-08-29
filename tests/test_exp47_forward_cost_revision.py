from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.forward_cost_revision_exp47 import (
    AUXILIARY_INPUT_FEATURES,
    EXP47_FEATURES,
    FORBIDDEN_AUX_INPUTS,
    MIN_REVISION_PP,
    build_forward_cost_revision_dataset,
)
from backend.app.ml.experiments.framework import experiment_run_directory
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import temporal_project_split


def _history() -> pd.DataFrame:
    rows = []
    for month, revised, spend in [(1, 100.0, 5.0), (2, 100.1, 8.0), (4, 110.0, 15.0), (8, 108.0, 30.0), (14, 125.0, 55.0)]:
        date = pd.Timestamp("2020-01-01") + pd.DateOffset(months=month - 1)
        rows.append({
            "canonical_project_id": "A", "snapshot_date": date, "approved_cost_cr": 100.0,
            "revised_cost_cr": revised, "cumulative_expenditure_cr": spend,
            "schedule_slippage_days": month * 3, "duration_ratio": month / 24,
            "sector": "Power", "project_size_category": "Medium", "current_schedule_status": "Delayed",
        })
    return pd.DataFrame(rows)


def test_future_append_does_not_change_earlier_auxiliary_inputs():
    history = _history()
    before = build_forward_cost_revision_dataset(history)
    future = pd.concat([history, pd.DataFrame([{
        **history.iloc[-1].to_dict(), "snapshot_date": "2025-01-01", "revised_cost_cr": 900.0,
    }])], ignore_index=True)
    after = build_forward_cost_revision_dataset(future)
    columns = ["canonical_project_id", "snapshot_date", *AUXILIARY_INPUT_FEATURES]
    pd.testing.assert_frame_equal(before[columns], after.iloc[: len(before)][columns])


def test_forward_labels_use_meaningful_events_and_preserve_censoring():
    frame = build_forward_cost_revision_dataset(_history())
    assert MIN_REVISION_PP == 0.25
    # The 0.1 pp parser-sized movement is ignored; the first real event is +9.9 pp.
    assert frame.loc[0, "cost_revision_within_3m"] == 1.0
    assert frame.loc[0, "next_cost_revision_pp"] > 9.0
    # No later report is not silently treated as no future revision.
    assert pd.isna(frame.iloc[-1]["cost_revision_within_3m"])
    assert frame.iloc[-1]["auxiliary_next_revision_observed"] == 0


def test_duplicate_monthly_rows_are_deterministic():
    history = pd.concat([_history(), _history().iloc[[2]].assign(revised_cost_cr=111.0)], ignore_index=True)
    first = build_forward_cost_revision_dataset(history)
    second = build_forward_cost_revision_dataset(history.sample(frac=1, random_state=47))
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["canonical_project_id", "snapshot_date"]).any()


def test_auxiliary_inputs_exclude_completion_and_final_outcomes():
    assert not (FORBIDDEN_AUX_INPUTS & set(AUXILIARY_INPUT_FEATURES))
    assert all("actual_" not in feature and "completion_date" not in feature for feature in AUXILIARY_INPUT_FEATURES)
    assert len(EXP47_FEATURES) == 6


def test_temporal_split_has_no_project_overlap():
    frame = pd.DataFrame({"canonical_project_id": ["A", "B"], "completion_year": [2019, 2022]})
    train, holdout = temporal_project_split(frame, 2001, 2019, 2025)
    assert set(train.canonical_project_id).isdisjoint(holdout.canonical_project_id)


def test_weights_and_ledger_match_final_cost_cohort():
    rows = assign_project_balanced_weights(pd.DataFrame({
        "canonical_project_id": ["A", "A", "B"],
        "snapshot_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]),
        "actual_cost_overrun_percentage": [10.0, 20.0, 30.0],
    }))
    assert np.allclose(rows.groupby("canonical_project_id").sample_weight.sum(), 1.0)
    ledger = build_prediction_ledger(
        rows, experiment_id="exp_47", window="2001_2019",
        production_cost_prediction=[9.0, 18.0, 35.0], experiment_cost_prediction=[10.0, 19.0, 32.0],
    )
    assert_prediction_ledger_matches_cohort(ledger, rows)
    assert (ledger.cost_abs_error_improvement == ledger.production_cost_abs_error - ledger.experiment_cost_abs_error).all()


def test_adapter_and_experiment_only_persistence_contract():
    assert get_experiment_adapter("exp_47").sequence == 47
    destination = experiment_run_directory("exp_47", "2001_2021", "test-run")
    assert "experiments/exp_47/2001_2021/test-run" in destination.as_posix()
    source = Path("backend/app/ml/experiments/forward_cost_revision_exp47.py").read_text()
    assert "joblib.dump" not in source
    assert "production_bundle[\"delay\"]" in source
    assert "candidate_delay = production_delay.copy()" in source
