import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from backend.app.ml.experiments.exp140_uncertainty_cost import (
    EXP140_RESIDUAL_FEATURES,
    QUANTILE_BASE_FEATURES,
    _fit_quantile_models,
    _stage_metrics,
    extract_uncertainty_features,
)


def test_extract_uncertainty_features_mathematical_consistency():
    n_samples = 50
    rng = np.random.RandomState(140)
    frame = pd.DataFrame(
        {
            "production_prediction": rng.uniform(10, 50, n_samples),
            "approved_cost_cr": rng.uniform(100, 1000, n_samples),
            "expenditure_ratio": rng.uniform(0.1, 1.2, n_samples),
            "cost_escalation_percentage": rng.uniform(0, 40, n_samples),
            "progress_deviation": rng.uniform(-20, 20, n_samples),
            "duration_ratio": rng.uniform(0.8, 2.0, n_samples),
            "physical_progress": rng.uniform(10, 100, n_samples),
            "actual_cost_overrun_percentage": rng.uniform(5, 60, n_samples),
            "sample_weight": np.ones(n_samples),
        }
    )

    models, cols, medians = _fit_quantile_models(frame, QUANTILE_BASE_FEATURES, seed=140, n_estimators=10)
    out = extract_uncertainty_features(frame, models, cols, medians)

    assert "exp140_interval_width" in out.columns
    assert "exp140_upper_asymmetry" in out.columns
    assert "exp140_lower_asymmetry" in out.columns
    assert "exp140_relative_uncertainty" in out.columns

    assert (out["exp140_interval_width"] >= 0.0).all()
    assert (out["exp140_upper_asymmetry"] >= 0.0).all()
    assert (out["exp140_lower_asymmetry"] >= 0.0).all()
    assert (out["exp140_relative_uncertainty"] >= 0.0).all()


def test_stage_metrics_returns_valid_structure():
    frame = pd.DataFrame(
        {
            "lifecycle_stage": ["early", "early", "mid", "mid", "late", "late", "very_late", "very_late"],
            "actual_cost_overrun_percentage": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0],
            "sample_weight": [1.0] * 8,
            "canonical_project_id": [f"proj_{i}" for i in range(8)],
        }
    )
    pred = np.array([9.0, 11.0, 14.0, 19.0, 21.0, 23.0, 24.0, 28.0])
    metrics = _stage_metrics(frame, pred)

    for stage in ("early", "mid", "late", "very_late"):
        assert stage in metrics
        assert metrics[stage]["available"] is True
        assert "MAE" in metrics[stage]
        assert "R2" in metrics[stage]
