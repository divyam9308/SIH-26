from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.cost_revision_dynamics_exp45 import (
    EXP45_FEATURES,
    FORBIDDEN_INPUTS,
    SOURCE_COLUMNS,
    engineer_revision_history,
)
from backend.app.ml.experiments.framework import experiment_run_directory
from backend.app.ml.experiments.prediction_ledger import (
    assert_prediction_ledger_matches_cohort,
    build_prediction_ledger,
)
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import temporal_project_split


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"canonical_project_id": "A", "snapshot_date": "2020-01-01", "approved_cost_cr": 100, "revised_cost_cr": 100},
            {"canonical_project_id": "A", "snapshot_date": "2020-04-01", "approved_cost_cr": 100, "revised_cost_cr": 110},
            {"canonical_project_id": "A", "snapshot_date": "2020-07-01", "approved_cost_cr": 100, "revised_cost_cr": 108},
            {"canonical_project_id": "A", "snapshot_date": "2020-10-01", "approved_cost_cr": 100, "revised_cost_cr": 125},
        ]
    )


def test_future_append_cannot_change_earlier_features_or_use_future_revised_cost():
    history = _history()
    before = engineer_revision_history(history)
    future = pd.concat(
        [history, pd.DataFrame([{"canonical_project_id": "A", "snapshot_date": "2022-01-01", "approved_cost_cr": 100, "revised_cost_cr": 500}])],
        ignore_index=True,
    )
    after = engineer_revision_history(future)
    pd.testing.assert_frame_equal(before, after.iloc[: len(before)].reset_index(drop=True))


def test_duplicate_observations_resolve_deterministically():
    duplicate = pd.concat([_history(), _history().iloc[[1]].assign(revised_cost_cr=111)], ignore_index=True)
    first = engineer_revision_history(duplicate)
    second = engineer_revision_history(duplicate.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["canonical_project_id", "snapshot_date"]).any()


def test_no_revision_has_stable_finite_fallbacks():
    frame = engineer_revision_history(_history().iloc[[0]])
    assert np.isfinite(frame[EXP45_FEATURES].to_numpy(float)).all()
    assert frame.loc[0, "exp45_revision_count_total"] == 0
    assert frame.loc[0, "exp45_months_since_last_revision"] == -1


def test_feature_contract_has_no_outcome_or_completion_inputs():
    assert not (FORBIDDEN_INPUTS & SOURCE_COLUMNS)
    assert all("completion" not in feature and "actual" not in feature for feature in EXP45_FEATURES)


def test_temporal_split_has_no_project_overlap():
    frame = pd.DataFrame(
        {"canonical_project_id": ["A", "B"], "completion_year": [2019, 2022], "sample_weight": [1.0, 1.0]}
    )
    train, holdout = temporal_project_split(frame, 2001, 2019, 2025)
    assert set(train.canonical_project_id).isdisjoint(holdout.canonical_project_id)


def test_weights_are_recomputed_after_filtering_and_ledger_matches_scored_cohort():
    rows = pd.DataFrame(
        {
            "canonical_project_id": ["A", "A", "B"],
            "snapshot_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01"]),
            "actual_cost_overrun_percentage": [10.0, 20.0, 30.0],
        }
    )
    rows = assign_project_balanced_weights(rows)
    assert np.allclose(rows.groupby("canonical_project_id").sample_weight.sum(), 1.0)
    ledger = build_prediction_ledger(
        rows,
        experiment_id="exp_45",
        window="2001_2019",
        production_cost_prediction=[9.0, 18.0, 35.0],
        experiment_cost_prediction=[10.0, 19.0, 32.0],
    )
    assert_prediction_ledger_matches_cohort(ledger, rows)


def test_adapter_discovery_and_experiment_only_artifact_path():
    assert get_experiment_adapter("exp_45").sequence == 45
    destination = experiment_run_directory("exp_45", "2001_2021", "test-run")
    assert "experiments/exp_45/2001_2021/test-run" in destination.as_posix()
    assert not destination.exists()
    source = Path("backend/app/ml/experiments/cost_revision_dynamics_exp45.py").read_text()
    assert "joblib.dump" not in source
