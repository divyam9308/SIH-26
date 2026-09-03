from pathlib import Path

import pandas as pd

from backend.app.ml.experiments import dataset_window_comparison as comparison


def test_window_contract_is_exact():
    assert comparison.WINDOWS == {
        "2001_2017": {"training_start": 2001, "training_end": 2017, "test_start": 2018, "test_end": 2025},
        "2001_2021": {"training_start": 2001, "training_end": 2021, "test_start": 2022, "test_end": 2025},
        "2001_2022": {"training_start": 2001, "training_end": 2022, "test_start": 2023, "test_end": 2025},
    }


def test_run_window_uses_current_production_signature(monkeypatch, tmp_path: Path):
    data = pd.DataFrame({"x": [1]})
    identity = pd.DataFrame({"row_index": [0], "identity_verified": [True]})
    observed = {}

    def fake_train(training_start, training_end, test_end, data=None, identity=None, artifact_root=None):
        observed.update(
            training_start=training_start,
            training_end=training_end,
            test_end=test_end,
            data=data,
            identity=identity,
            artifact_root=artifact_root,
        )
        return {
            "lifecycle": {
                "metrics": {
                    "cost": {"MAE": 12.5, "RMSE": 20.0, "unique_projects": 7, "rows": 70},
                    "delay": {"MAE": 345.0, "RMSE": 500.0, "unique_projects": 7, "rows": 70},
                }
            },
            "metadata": {
                "production_cost_baseline": "cost-prod",
                "production_delay_baseline": "delay-prod",
                "cost_evaluation_contract": {"test_projects": 7, "test_snapshots": 70},
                "delay_evaluation_contract": {
                    "routing_policy": "all_evidence_eligible",
                    "aft_eligible_projects": 6,
                    "fallback_projects": 1,
                },
            },
        }

    monkeypatch.setattr(comparison, "train_window_with_promoted_cost_and_delay", fake_train)
    result = comparison.run_window(
        "2001_2017",
        data=data,
        identity=identity,
        artifact_root=tmp_path,
    )

    assert observed["training_start"] == 2001
    assert observed["training_end"] == 2017
    assert observed["test_end"] == 2025
    assert observed["data"] is data
    assert observed["identity"] is identity
    assert observed["artifact_root"] == tmp_path
    assert result["cost_mae"] == 12.5
    assert result["delay_mae_days"] == 345.0
    assert result["comparison_projects"] == 7
    assert result["comparison_snapshots"] == 70
    assert result["delay_aft_projects"] == 6
    assert result["delay_fallback_projects"] == 1
    assert result["production_main_contract"].startswith("PR186")


def test_unknown_window_rejected():
    try:
        comparison.get_window("2001_2020")
    except ValueError as exc:
        assert "Unknown dataset window" in str(exc)
    else:
        raise AssertionError("Unknown comparison window should be rejected")
