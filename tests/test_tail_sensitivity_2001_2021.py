import pandas as pd

from scripts.run_tail_sensitivity_2001_2021 import build_report, derive_training_thresholds


def _training_frame():
    rows = []
    for idx in range(1, 101):
        rows.append({
            "canonical_project_id": f"T{idx}",
            "completion_year": 2010,
            "actual_cost_overrun_percentage": float(idx),
            "actual_delay_days": float(idx * 10),
        })
    return pd.DataFrame(rows)


def _ledger():
    return pd.DataFrame([
        {
            "canonical_project_id": "A",
            "actual_cost_overrun_percentage": 10.0,
            "predicted_cost_overrun": 12.0,
            "actual_delay_days": 100.0,
            "predicted_delay_days": 120.0,
            "sample_weight": 1.0,
        },
        {
            "canonical_project_id": "B",
            "actual_cost_overrun_percentage": 95.0,
            "predicted_cost_overrun": 50.0,
            "actual_delay_days": 950.0,
            "predicted_delay_days": 500.0,
            "sample_weight": 1.0,
        },
        {
            "canonical_project_id": "C",
            "actual_cost_overrun_percentage": 100.0,
            "predicted_cost_overrun": 10.0,
            "actual_delay_days": 1000.0,
            "predicted_delay_days": 100.0,
            "sample_weight": 1.0,
        },
    ])


def test_thresholds_are_derived_from_training_projects_only():
    thresholds = derive_training_thresholds(_training_frame())
    assert thresholds["cost"]["p90"] == 90.1
    assert thresholds["delay"]["p95"] == 950.5
    assert thresholds["cost"]["source"].startswith("2001-2021")


def test_report_keeps_full_cohort_and_tail_diagnostics_separate():
    report = build_report(_ledger(), _training_frame())
    assert report["model_role"] == "diagnostic_only"
    assert report["interpretation"]["promotion_allowed"] is False
    assert report["cost"]["bands"]["all"]["unique_projects"] == 3
    assert report["cost"]["bands"]["excluding_top_5pct_le_p95"]["unique_projects"] == 2
    assert report["delay"]["bands"]["tail_gt_p95"]["unique_projects"] == 1
    assert report["window"] == {
        "training_start": 2001,
        "training_end": 2021,
        "test_start": 2022,
        "test_end": 2025,
    }


def test_excluding_extreme_tail_can_reduce_error_without_redefining_primary_metric():
    report = build_report(_ledger(), _training_frame())
    full = report["cost"]["bands"]["all"]
    trimmed = report["cost"]["bands"]["excluding_top_1pct_le_p99"]
    assert trimmed["MAE"] < full["MAE"]
    assert report["leakage_policy"].startswith("Tail thresholds are derived only from")
