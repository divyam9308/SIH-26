from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.experiments.framework import (
    assert_same_comparison_context,
    build_experiment_context,
    experiment_run_directory,
    new_experiment_manifest,
    promotion_guard,
)
from backend.app.ml.experiments.registry import load_registry, record_experiment
from backend.app.services.lifecycle_retraining_service import _stamp_production_role
from backend.app.services.lifecycle_run_service import lifecycle_runs


def _cohorts():
    full = pd.DataFrame([
        {"canonical_project_id": "A", "snapshot_date": "2014-01-01", "completion_year": 2014, "x": 1.0},
        {"canonical_project_id": "B", "snapshot_date": "2018-01-01", "completion_year": 2018, "x": 2.0},
    ])
    train = full.iloc[[0]].copy()
    test = full.iloc[[1]].copy()
    return full, train, test


def test_context_freezes_comparison_evidence_and_experiment_paths_are_run_scoped():
    full, train, test = _cohorts()
    context = build_experiment_context(
        experiment_id="exp_next",
        full_data=full,
        train=train,
        test=test,
        features=["x"],
        training_start=2001,
        training_end=2015,
        testing_end=2025,
        weighting_policy="per-project-post-sampling",
    )
    assert context.testing_start == 2016
    assert context.training_fingerprint != context.test_fingerprint
    manifest = new_experiment_manifest(
        context=context,
        name="candidate hypothesis",
        changed_dimension="algorithm",
        hypothesis="change one algorithm only",
        run_id="run-123",
    )
    assert manifest["model_role"] == "experiment"
    assert manifest["promotion_allowed"] is False
    path = experiment_run_directory("exp_next", context.window, manifest["run_id"])
    assert path.parts[-3:] == ("exp_next", "2001_2015", "run-123")


def test_comparison_contract_rejects_changed_test_or_weighting_context():
    baseline = {
        "training_fingerprint": "train-a",
        "test_fingerprint": "test-a",
        "feature_schema_fingerprint": "features-a",
        "weighting_policy": "per-project",
    }
    candidate = dict(baseline)
    assert_same_comparison_context(baseline, candidate)
    candidate["test_fingerprint"] = "test-b"
    with pytest.raises(ValueError, match="test_fingerprint"):
        assert_same_comparison_context(baseline, candidate)


def test_promotion_is_explicit_and_never_implied_by_training():
    manifest = {"model_role": "experiment", "decision": "PENDING", "promotion_allowed": False}
    with pytest.raises(PermissionError):
        promotion_guard(manifest)
    manifest.update(decision="ACCEPTED", promotion_allowed=True)
    promotion_guard(manifest)


def test_registry_is_atomic_canonical_and_deduplicates_run_id(tmp_path: Path):
    path = tmp_path / "registry.json"
    entry = {
        "experiment_id": "exp_next",
        "name": "candidate",
        "run_id": "run-1",
        "status": "COMPLETED",
        "decision": "PENDING",
    }
    record_experiment(entry, path=path)
    record_experiment({**entry, "decision": "REJECTED"}, path=path)
    payload = load_registry(path)
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["decision"] == "REJECTED"
    assert payload["production_policy"]["experiments_are_never_auto_promoted"] is True
    assert not path.with_suffix(".json.tmp").exists()


def test_production_registry_excludes_explicit_experiment_role(tmp_path: Path):
    run = tmp_path / "monthly_lifecycle" / "2001_2015"
    run.mkdir(parents=True)
    (run / "evaluation_results.json").write_text(json.dumps({
        "metadata": {
            "model_role": "experiment",
            "training_period": [2001, 2015],
            "testing_period": [2016, 2025],
        },
        "lifecycle": {"metrics": {}},
    }))
    registry = lifecycle_runs(tmp_path)
    assert registry == {"items": [], "count": 0}


def test_production_retrain_stamp_marks_both_persisted_truth_sources(tmp_path: Path):
    target = tmp_path / "2001_2015"
    target.mkdir()
    (target / "metadata.json").write_text(json.dumps({"model_version": "monthly-2001-2015"}))
    (target / "evaluation_results.json").write_text(json.dumps({"metadata": {"model_version": "monthly-2001-2015"}}))
    result = {"metadata": {"model_version": "monthly-2001-2015"}}
    _stamp_production_role(result, target)
    assert result["metadata"]["model_role"] == "production"
    assert json.loads((target / "metadata.json").read_text())["model_role"] == "production"
    assert json.loads((target / "evaluation_results.json").read_text())["metadata"]["model_role"] == "production"


def test_judge_facing_model_simulation_uses_controlled_orchestration_with_production_fallback():
    root = Path(__file__).resolve().parents[1]
    simulation = (root / "frontend" / "src" / "pages" / "ModelSimulationPage.js").read_text()
    accuracy = (root / "frontend" / "src" / "pages" / "PredictionAccuracyPage.js").read_text()
    api_source = (root / "frontend" / "src" / "services" / "api.js").read_text()

    # Prediction Accuracy remains production-only. Model Simulation may compare,
    # but it must use controlled server-side orchestration rather than invoking
    # an experiment directly from the browser. When no adapter is installed,
    # the selected year range must still support fresh production retraining and
    # a production-only judge session.
    assert "residualOverrunExperiment(" not in simulation
    assert "residualOverrunExperiment(" not in accuracy
    assert "api.retrainAndCompare(" in simulation
    assert "api.predictComparison(" in simulation
    assert "api.revealComparison(" in simulation
    assert "api.retrainModel(" in simulation
    assert "api.trainCustomSimulation(" in simulation
    assert "api.predictCustomSimulation(" in simulation
    assert "api.revealCustomSimulation(" in simulation
    assert "No challenger installed." in simulation
    assert "Retrain Production Model" in simulation
    assert "retrainAndCompare:" in api_source
    assert "/api/model-simulations/custom/retrain-compare" in api_source
