import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.residual_calibration_exp33_on_exp34 import (
    EXPERIMENT_ID,
    EXPERIMENT_SCOPE,
    _corrections,
    _fit_residual_calibration,
    _weighted_median,
)


def test_adapter_is_registered_as_delay_only_ablation():
    adapter = get_experiment_adapter(EXPERIMENT_ID)
    assert adapter.experiment_id == "exp_33_on_exp34"
    assert adapter.scope == "delay"
    assert EXPERIMENT_SCOPE == "delay"


def test_weighted_median_respects_project_balanced_weights():
    assert _weighted_median([1.0, 5.0, 9.0], [1.0, 5.0, 1.0]) == 5.0


def test_calibration_uses_stage_bin_then_fallback_without_nonfinite_public_edges():
    rows = []
    for index in range(40):
        prediction = float(index)
        stage = "early" if index < 20 else "late"
        residual = 10.0 if stage == "early" else -5.0
        rows.append(
            {
                "prediction": prediction,
                "residual": residual,
                "sample_weight": 1.0,
                "lifecycle_stage": stage,
            }
        )
    oof = pd.DataFrame(rows)
    calibration = _fit_residual_calibration(oof)
    frame = pd.DataFrame(
        {
            "lifecycle_stage": ["early", "late", "unknown"],
        }
    )
    prediction = np.asarray([2.0, 35.0, 18.0])
    correction = _corrections(frame, prediction, calibration)

    assert np.isfinite(correction).all()
    assert correction[0] >= 0
    assert correction[1] <= 0
    assert len(calibration["edges"]) >= 3
    assert np.isneginf(calibration["edges"][0])
    assert np.isposinf(calibration["edges"][-1])
