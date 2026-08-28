import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp35_aft_residual_combo import (
    EXPERIMENT_ID,
    EXPERIMENT_SEQUENCE,
    _corrections,
    _delay_from_remaining,
    _fit_residual_calibration,
    _weighted_median,
)


def test_exp35_identity():
    assert EXPERIMENT_ID == "exp_35"
    assert EXPERIMENT_SEQUENCE == 35


def test_weighted_median_respects_weights():
    assert _weighted_median([0, 10, 20], [1, 10, 1]) == 10.0


def test_delay_from_remaining_uses_asof_planned_completion():
    frame = pd.DataFrame([{
        "snapshot_date": "2024-01-01",
        "planned_completion_date": "2024-01-21",
    }])
    delay = _delay_from_remaining(frame, np.asarray([30.0]))
    assert round(float(delay[0]), 6) == 10.0


def test_residual_calibration_is_finite_and_applicable():
    oof = pd.DataFrame({
        "prediction": np.linspace(0, 99, 100),
        "residual": np.where(np.arange(100) < 50, 5.0, -3.0),
        "sample_weight": np.ones(100),
        "lifecycle_stage": ["early"] * 50 + ["late"] * 50,
    })
    calibration = _fit_residual_calibration(oof)
    frame = pd.DataFrame({"lifecycle_stage": ["early", "late"]})
    corr = _corrections(frame, np.asarray([10.0, 90.0]), calibration)
    assert np.isfinite(corr).all()
    assert corr[0] == 5.0
    assert corr[1] == -3.0
