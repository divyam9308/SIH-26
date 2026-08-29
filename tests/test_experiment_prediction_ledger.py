from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.experiments.prediction_ledger import (
    LEDGER_FILENAME,
    LEDGER_MANIFEST_FILENAME,
    assert_prediction_ledger_matches_cohort,
    build_prediction_ledger,
    cohort_fingerprint,
    validate_prediction_ledger,
    write_prediction_ledger,
)


def _rows() -> pd.DataFrame:
    # Deliberately unsorted so the test catches prediction/row misalignment.
    return pd.DataFrame([
        {
            "canonical_project_id": "B",
            "snapshot_date": "2024-01-01",
            "sample_weight": 1.0,
            "actual_cost_overrun_percentage": 30.0,
            "actual_delay_days": 300.0,
            "lifecycle_stage": "late",
            "sector": "Power",
        },
        {
            "canonical_project_id": "A",
            "snapshot_date": "2024-04-01",
            "sample_weight": 0.5,
            "actual_cost_overrun_percentage": 20.0,
            "actual_delay_days": 200.0,
            "lifecycle_stage": "mid",
            "sector": "Roads",
        },
        {
            "canonical_project_id": "A",
            "snapshot_date": "2024-01-01",
            "sample_weight": 0.5,
            "actual_cost_overrun_percentage": 10.0,
            "actual_delay_days": 100.0,
            "lifecycle_stage": "early",
            "sector": "Roads",
        },
    ])


def test_prediction_ledger_preserves_prediction_alignment_and_computes_row_errors():
    rows = _rows()
    ledger = build_prediction_ledger(
        rows,
        experiment_id="exp_45",
        window="2001_2021",
        # Values correspond to incoming B, A-late, A-early order.
        production_cost_prediction=[35.0, 18.0, 14.0],
        experiment_cost_prediction=[32.0, 19.0, 11.0],
        production_delay_prediction=[330.0, 220.0, 130.0],
        experiment_delay_prediction=[310.0, 205.0, 110.0],
    )

    assert list(zip(ledger.canonical_project_id, ledger.snapshot_date.dt.strftime("%Y-%m-%d"))) == [
        ("A", "2024-01-01"),
        ("A", "2024-04-01"),
        ("B", "2024-01-01"),
    ]
    assert ledger.production_cost_prediction.tolist() == [14.0, 18.0, 35.0]
    assert ledger.experiment_cost_prediction.tolist() == [11.0, 19.0, 32.0]
    assert ledger.cost_abs_error_improvement.tolist() == [3.0, 1.0, 3.0]
    assert ledger.production_delay_prediction.tolist() == [130.0, 220.0, 330.0]
    assert ledger.experiment_delay_prediction.tolist() == [110.0, 205.0, 310.0]
    assert ledger.delay_abs_error_improvement.tolist() == [20.0, 15.0, 20.0]
    assert ledger.lifecycle_stage.tolist() == ["early", "mid", "late"]

    diagnostics = validate_prediction_ledger(ledger)
    assert diagnostics["projects"] == 2
    assert diagnostics["snapshots"] == 3
    assert diagnostics["targets"] == ["cost", "delay"]
    assert diagnostics["project_weight_sum_min"] == pytest.approx(1.0)
    assert diagnostics["project_weight_sum_max"] == pytest.approx(1.0)
    assert_prediction_ledger_matches_cohort(ledger, rows)


def test_target_specific_ledger_is_valid_and_does_not_require_fabricated_other_target():
    rows = _rows()
    ledger = build_prediction_ledger(
        rows,
        experiment_id="exp_46",
        window="2001_2021",
        production_delay_prediction=[330.0, 220.0, 130.0],
        experiment_delay_prediction=[310.0, 205.0, 110.0],
    )
    diagnostics = validate_prediction_ledger(ledger)
    assert diagnostics["targets"] == ["delay"]
    assert "production_cost_prediction" not in ledger
    assert "experiment_cost_prediction" not in ledger


def test_prediction_ledger_rejects_duplicate_keys_and_unbalanced_project_weights():
    rows = _rows()
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate project/snapshot"):
        build_prediction_ledger(
            duplicate,
            experiment_id="exp_bad",
            window="2001_2021",
            production_cost_prediction=[1.0, 2.0, 3.0, 4.0],
            experiment_cost_prediction=[1.0, 2.0, 3.0, 4.0],
        )

    unbalanced = rows.copy()
    unbalanced.loc[unbalanced.canonical_project_id == "A", "sample_weight"] = 0.25
    with pytest.raises(ValueError, match="each project has total weight 1"):
        build_prediction_ledger(
            unbalanced,
            experiment_id="exp_bad",
            window="2001_2021",
            production_cost_prediction=[35.0, 18.0, 14.0],
            experiment_cost_prediction=[32.0, 19.0, 11.0],
        )


def test_prediction_ledger_cohort_fingerprint_changes_with_post_filter_weighting():
    rows = _rows()
    changed = rows.copy()
    changed.loc[changed.canonical_project_id == "A", "sample_weight"] = [0.75, 0.25]
    assert cohort_fingerprint(rows) != cohort_fingerprint(changed)


def test_prediction_ledger_persistence_is_immutable_and_manifest_is_self_describing(tmp_path: Path):
    rows = _rows()
    ledger = build_prediction_ledger(
        rows,
        experiment_id="exp_45",
        window="2001_2021",
        production_cost_prediction=[35.0, 18.0, 14.0],
        experiment_cost_prediction=[32.0, 19.0, 11.0],
    )
    result = write_prediction_ledger(
        ledger,
        tmp_path / "run-1",
        extra_manifest={"primary_target": "cost", "execution_verdict": "EXECUTION VALID"},
    )

    assert result["ledger_path"].name == LEDGER_FILENAME
    assert result["manifest_path"].name == LEDGER_MANIFEST_FILENAME
    manifest = json.loads(result["manifest_path"].read_text())
    assert manifest["schema_version"] == 1
    assert manifest["experiment_id"] == "exp_45"
    assert manifest["window"] == "2001_2021"
    assert manifest["projects"] == 2
    assert manifest["snapshots"] == 3
    assert manifest["targets"] == ["cost"]
    assert manifest["primary_target"] == "cost"
    assert manifest["execution_verdict"] == "EXECUTION VALID"
    assert manifest["ledger_file_sha256"]
    assert manifest["metrics"]["cost"]["experiment_mae"] < manifest["metrics"]["cost"]["production_mae"]

    with pytest.raises(FileExistsError, match="immutable once written"):
        write_prediction_ledger(ledger, tmp_path / "run-1")


def test_prediction_ledger_rejects_a_different_scored_cohort():
    rows = _rows()
    ledger = build_prediction_ledger(
        rows,
        experiment_id="exp_45",
        window="2001_2021",
        production_cost_prediction=[35.0, 18.0, 14.0],
        experiment_cost_prediction=[32.0, 19.0, 11.0],
    )
    different = rows.iloc[1:].copy()
    different.loc[different.canonical_project_id == "A", "sample_weight"] = 0.5
    with pytest.raises(ValueError, match="cohort mismatch"):
        assert_prediction_ledger_matches_cohort(ledger, different)
