from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.framework import experiment_run_directory
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger
from backend.app.ml.experiments.spend_revision_leadlag_exp48 import EXP48_FEATURES, FORBIDDEN_INPUTS, LAGS_MONTHS, SOURCE_COLUMNS, engineer_spend_revision_leadlag
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import temporal_project_split


def _history() -> pd.DataFrame:
    return pd.DataFrame([{"canonical_project_id": "A", "snapshot_date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=i), "approved_cost_cr": 100.0, "revised_cost_cr": revised, "cumulative_expenditure_cr": spend, "duration_ratio": (i + 1) / 12} for i, (spend, revised) in enumerate([(5, 100), (8, 100), (15, 100), (25, 110), (32, 110), (39, 110), (50, 125), (59, 125)])])


def test_future_append_cannot_change_earlier_leadlag_features():
    history = _history(); before = engineer_spend_revision_leadlag(history)
    future = pd.concat([history, pd.DataFrame([{**history.iloc[-1].to_dict(), "snapshot_date": "2024-01-01", "revised_cost_cr": 900, "cumulative_expenditure_cr": 800}])], ignore_index=True)
    after = engineer_spend_revision_leadlag(future)
    pd.testing.assert_frame_equal(before, after.iloc[: len(before)].reset_index(drop=True))


def test_correlations_are_fixed_past_lags_and_finite():
    frame = engineer_spend_revision_leadlag(_history())
    assert LAGS_MONTHS == (1, 3, 6)
    correlation = [name for name in EXP48_FEATURES if "corr_lag" in name]
    assert len(correlation) == 6
    assert np.isfinite(frame[EXP48_FEATURES].to_numpy(float)).all()
    assert (frame[correlation].abs() <= 1.0 + 1e-12).all().all()


def test_duplicates_resolve_deterministically_and_no_future_cost_input():
    history = pd.concat([_history(), _history().iloc[[3]].assign(revised_cost_cr=111)], ignore_index=True)
    first = engineer_spend_revision_leadlag(history); second = engineer_spend_revision_leadlag(history.sample(frac=1, random_state=48))
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["canonical_project_id", "snapshot_date"]).any()
    assert not (FORBIDDEN_INPUTS & SOURCE_COLUMNS)
    assert all("actual" not in feature and "completion" not in feature for feature in EXP48_FEATURES)


def test_temporal_split_and_project_balanced_ledger_contract():
    train, holdout = temporal_project_split(pd.DataFrame({"canonical_project_id": ["A", "B"], "completion_year": [2019, 2022]}), 2001, 2019, 2025)
    assert set(train.canonical_project_id).isdisjoint(holdout.canonical_project_id)
    rows = assign_project_balanced_weights(pd.DataFrame({"canonical_project_id": ["A", "A", "B"], "snapshot_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]), "actual_cost_overrun_percentage": [10.0, 20.0, 30.0]}))
    assert np.allclose(rows.groupby("canonical_project_id").sample_weight.sum(), 1.0)
    ledger = build_prediction_ledger(rows, experiment_id="exp_48", window="2001_2019", production_cost_prediction=[9, 18, 35], experiment_cost_prediction=[10, 19, 32])
    assert_prediction_ledger_matches_cohort(ledger, rows)


def test_no_exp47_dependency_adapter_and_experiment_only_persistence():
    assert get_experiment_adapter("exp_48").sequence == 48
    destination = experiment_run_directory("exp_48", "2001_2021", "test-run")
    assert "experiments/exp_48/2001_2021/test-run" in destination.as_posix()
    source = Path("backend/app/ml/experiments/spend_revision_leadlag_exp48.py").read_text()
    assert "exp47_" not in source.lower()
    assert "joblib.dump" not in source
    assert "candidate_delay = production_delay.copy()" in source
