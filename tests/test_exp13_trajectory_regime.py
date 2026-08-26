from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.adapters import default_experiment_adapter, get_experiment_adapter
from backend.app.ml.experiments.trajectory_exp12_v2 import enrich_rows as enrich_exp12_rows
from backend.app.ml.experiments.trajectory_exp13 import (
    CROSS_SIGNAL_FEATURES,
    EXP13_FEATURES,
    LIFECYCLE_FEATURES,
    REGIME_FEATURES,
    TRANSITION_FEATURES,
    TURNING_FEATURES,
    _candidate_groups,
    engineer_history,
    enrich_rows,
)


def _history() -> pd.DataFrame:
    rows = []
    revised = [100, 100, 102, 105, 112, 122, 136, 153]
    spend = [8, 15, 23, 31, 40, 50, 61, 73]
    slip = [0, 0, 5, 12, 28, 52, 85, 125]
    dates = pd.to_datetime(
        [
            "2021-01-31",
            "2021-02-28",
            "2021-03-31",
            "2021-04-30",
            "2021-05-31",
            "2021-06-30",
            "2021-07-31",
            "2021-08-31",
        ]
    )
    for i, stamp in enumerate(dates):
        rows.append(
            {
                "canonical_project_id": "P",
                "snapshot_date": stamp,
                "approved_cost_cr": 100.0,
                "revised_cost_cr": float(revised[i]),
                "cumulative_expenditure_cr": float(spend[i]),
                "schedule_slippage_days": float(slip[i]),
                "planned_duration_days": 730.0,
                "expected_progress_percentage": float(10 + i * 10),
                "planned_completion_date": "2022-12-31",
                "revised_completion_date": "2022-12-31" if i < 4 else "2023-06-30",
            }
        )
    return pd.DataFrame(rows)


def test_exp13_history_features_are_as_of_safe_when_future_reports_are_added():
    history = _history()
    before = engineer_history(history)
    future = history.iloc[-1].copy()
    future["snapshot_date"] = pd.Timestamp("2021-09-30")
    future["revised_cost_cr"] = 9999.0
    future["cumulative_expenditure_cr"] = 9999.0
    future["schedule_slippage_days"] = 9999.0
    extended = pd.concat([history, pd.DataFrame([future])], ignore_index=True)
    after = engineer_history(extended)

    cutoff = pd.Timestamp("2021-08-31")
    columns = REGIME_FEATURES + CROSS_SIGNAL_FEATURES + TURNING_FEATURES + TRANSITION_FEATURES
    before_rows = before[pd.to_datetime(before.snapshot_date).le(cutoff)].reset_index(drop=True)
    after_rows = after[pd.to_datetime(after.snapshot_date).le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_rows[columns], after_rows[columns])


def test_exp13_enrichment_preserves_cohort_and_adds_lifecycle_interactions():
    history = _history()
    supervised = history.iloc[[3, 7]].copy()
    supervised["completion_year"] = [2023, 2023]
    supervised["sample_weight"] = [0.5, 0.5]
    supervised["lifecycle_stage"] = ["mid", "very_late"]
    production_enriched = enrich_exp12_rows(supervised, history)

    enriched = enrich_rows(production_enriched, history)
    assert len(enriched) == len(supervised)
    assert enriched.canonical_project_id.tolist() == supervised.canonical_project_id.tolist()
    assert set(EXP13_FEATURES).issubset(enriched.columns)
    assert enriched.exp13_lifecycle_progress.tolist() == [0.375, 0.875]
    assert enriched.loc[enriched.index[-1], "exp13_compound_pressure_score"] >= 0
    assert enriched[LIFECYCLE_FEATURES].notna().any(axis=1).all()


def test_exp13_candidate_groups_include_safe_production_fallback():
    groups = _candidate_groups(EXP13_FEATURES)
    assert groups["production_only"] == []
    assert set(REGIME_FEATURES).issubset(groups["regime_scores"])
    assert set(TRANSITION_FEATURES).issubset(groups["all_regime_context"])
    assert len(groups["all_regime_context"]) >= len(groups["regime_plus_interactions"])


def test_exp13_adapter_is_registered_as_default_challenger():
    adapter = get_experiment_adapter("exp_13")
    assert adapter.sequence == 13
    assert adapter.scope == "cost_delay"
    assert default_experiment_adapter().experiment_id == "exp_13"
