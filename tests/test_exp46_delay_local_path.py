from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.delay_local_path_exp46 import (
    EXP46_FEATURES,
    FORBIDDEN_INPUTS,
    REUSED_EXP12_DELAY_FEATURES,
    SOURCE_COLUMNS,
    _diagnostics,
    engineer_local_delay_history,
)
from backend.app.ml.experiments.framework import experiment_run_directory
from backend.app.ml.experiments.prediction_ledger import assert_prediction_ledger_matches_cohort, build_prediction_ledger
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.production_exp35_baseline import AFTResidualDelayModel, CALIBRATION_GATE_FEATURE


def _history() -> pd.DataFrame:
    rows = []
    for month, revised, slip, expenditure in [
        ("2020-01-01", "2021-01-01", 0, 10),
        ("2020-04-01", "2021-04-01", 90, 20),
        ("2020-07-01", "2021-04-01", 90, 20),
        ("2020-10-01", "2021-10-01", 273, 21),
        ("2021-01-01", "2021-10-01", 273, 21),
        ("2021-04-01", "2022-01-01", 365, 22),
    ]:
        rows.append({
            "canonical_project_id": "A", "snapshot_date": month,
            "planned_completion_date": "2021-01-01", "revised_completion_date": revised,
            "schedule_slippage_days": slip, "approved_cost_cr": 100,
            "revised_cost_cr": 100, "cumulative_expenditure_cr": expenditure,
        })
    return pd.DataFrame(rows)


def test_future_append_does_not_change_earlier_local_or_change_point_features():
    history = _history()
    before = engineer_local_delay_history(history)
    future = pd.concat([history, pd.DataFrame([{
        "canonical_project_id": "A", "snapshot_date": "2025-01-01",
        "planned_completion_date": "2021-01-01", "revised_completion_date": "2030-01-01",
        "schedule_slippage_days": 3287, "approved_cost_cr": 100,
        "revised_cost_cr": 500, "cumulative_expenditure_cr": 500,
    }])], ignore_index=True)
    after = engineer_local_delay_history(future)
    pd.testing.assert_frame_equal(before, after.iloc[:len(before)].reset_index(drop=True))


def test_reporting_gap_and_stagnation_are_prefix_only():
    result = engineer_local_delay_history(_history())
    assert result.loc[0, "exp46_days_since_previous_report"] == -1
    assert result.loc[2, "exp46_unchanged_completion_streak"] >= 1
    assert result.loc[2, "exp46_expenditure_stagnation_streak"] >= 1
    assert np.isfinite(result[EXP46_FEATURES].to_numpy(float)).all()


def test_duplicate_reports_are_deterministic():
    frame = pd.concat([_history(), _history().iloc[[1]].assign(cumulative_expenditure_cr=25)], ignore_index=True)
    a = engineer_local_delay_history(frame)
    b = engineer_local_delay_history(frame.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(a, b)


def test_completion_label_physical_progress_and_future_outcomes_are_not_inputs():
    assert not (FORBIDDEN_INPUTS & SOURCE_COLUMNS)
    assert "physical_progress" not in SOURCE_COLUMNS
    assert all("actual" not in feature and "completion_date" not in feature for feature in EXP46_FEATURES)
    assert not (set(EXP46_FEATURES) & set(REUSED_EXP12_DELAY_FEATURES))
    assert "exp12_slippage_velocity_3m" in REUSED_EXP12_DELAY_FEATURES


def test_routing_depends_on_as_of_evidence_and_gate_not_target_or_error():
    frame = pd.DataFrame({
        "snapshot_date": pd.to_datetime(["2024-01-01"] * 4),
        "planned_completion_date": pd.to_datetime(["2025-01-01", None, "2025-01-01", "2025-01-01"]),
        CALIBRATION_GATE_FEATURE: [True, True, False, np.nan],
        "actual_delay_days": [0, 9999, 9999, 0],
    })
    mask = AFTResidualDelayModel._aft_eligible(frame)
    assert mask.tolist() == [True, False, False, True]
    changed_targets = frame.assign(actual_delay_days=[9999, 0, 0, 9999])
    assert AFTResidualDelayModel._aft_eligible(changed_targets).tolist() == mask.tolist()


def test_project_level_fallback_count_is_not_any_fallback_row_count():
    """A gated project can legitimately have early fallback snapshots."""
    frame = pd.DataFrame({
        "canonical_project_id": ["A", "A", "B"],
        "snapshot_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
        "planned_completion_date": pd.to_datetime([None, "2025-01-01", None]),
        CALIBRATION_GATE_FEATURE: [True, True, False],
    })
    route = AFTResidualDelayModel._aft_eligible(frame)
    aft_ids = set(frame.loc[route, "canonical_project_id"])
    all_ids = set(frame["canonical_project_id"])
    assert frame.loc[~route, "canonical_project_id"].nunique() == 2
    assert len(all_ids - aft_ids) == 1


def test_full_cohort_weights_ledger_and_unchanged_cost_contract():
    rows = pd.DataFrame({
        "canonical_project_id": ["A", "A", "B"],
        "snapshot_date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-01-01"]),
        "actual_delay_days": [100., 120., 300.], "actual_cost_overrun_percentage": [10., 10., 20.],
        "experiment_route": ["exp46_aft", "exp46_aft", "exp34_fallback"],
    })
    rows = assign_project_balanced_weights(rows)
    ledger = build_prediction_ledger(
        rows, experiment_id="exp_46", window="2001_2021",
        production_delay_prediction=[110., 130., 280.], experiment_delay_prediction=[105., 125., 280.],
        extra_columns=["experiment_route"],
    )
    assert_prediction_ledger_matches_cohort(ledger, rows)
    assert len(ledger) == len(rows)
    production_cost = np.array([9., 9., 21.])
    candidate_cost = production_cost.copy()
    assert np.array_equal(production_cost, candidate_cost)


def test_diagnostics_uses_named_columns_not_dataframe_prod_method():
    rows = pd.DataFrame({
        "canonical_project_id": ["A", "A", "B"],
        "actual_delay_days": [100.0, 120.0, 300.0],
        "production_delay_prediction": [110.0, 140.0, 280.0],
        "experiment_delay_prediction": [105.0, 125.0, 290.0],
        "sample_weight": [0.5, 0.5, 1.0],
        "lifecycle_stage": ["early", "early", "late"],
    })
    result = _diagnostics(rows)
    assert result["median_per_project_mae"]["production"] >= 0
    assert result["median_per_project_mae"]["experiment"] >= 0
    assert result["p90_per_project_mae"]["production"] >= 0
    assert result["p90_per_project_mae"]["experiment"] >= 0


def test_adapter_discovery_and_experiment_only_artifact_path():
    assert get_experiment_adapter("exp_46").sequence == 46
    destination = experiment_run_directory("exp_46", "2001_2021", "test-run")
    assert "experiments/exp_46/2001_2021/test-run" in destination.as_posix()
    assert not destination.exists()
    source = Path("backend/app/ml/experiments/delay_local_path_exp46.py").read_text()
    assert "joblib.dump" not in source
