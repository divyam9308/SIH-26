import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import FAMILIES
from backend.app.ml.production_exp35_baseline import (
    AFTResidualDelayModel,
    ResidualCalibratedCostModel,
    PRODUCTION_COST_BASELINE,
    PRODUCTION_DELAY_BASELINE,
)


class _ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def _calibration(value=0.0):
    return {
        "edges": [-np.inf, np.inf],
        "global_median": float(value),
        "bin_medians": {0: float(value)},
        "stage_bin_medians": {},
        "oof_rows": 100,
    }


def test_cost_wrapper_applies_exp33_residual_calibration():
    model = ResidualCalibratedCostModel(
        _ConstantModel(10.0), ["x"], _calibration(2.5)
    )
    frame = pd.DataFrame({"x": [1.0, 2.0], "lifecycle_stage": ["early", "late"]})
    np.testing.assert_allclose(model.predict(frame), [12.5, 12.5])


def test_delay_wrapper_preserves_exp34_fallback_without_planned_completion():
    aft_models = {family: _ConstantModel(np.log1p(10.0)) for family in FAMILIES}
    weights = {family: (1.0 if family == FAMILIES[0] else 0.0) for family in FAMILIES}
    model = AFTResidualDelayModel(
        aft_models=aft_models,
        weights=weights,
        features=["x"],
        calibration=_calibration(0.0),
        fallback_model=_ConstantModel(123.0),
    )
    frame = pd.DataFrame(
        {
            "x": [1.0],
            "snapshot_date": ["2020-01-01"],
            "planned_completion_date": [None],
            "lifecycle_stage": ["mid"],
        }
    )
    np.testing.assert_allclose(model.predict(frame), [123.0])


def test_delay_wrapper_uses_exp32_aft_then_exp33_when_evidence_exists():
    aft_models = {family: _ConstantModel(np.log1p(10.0)) for family in FAMILIES}
    weights = {family: (1.0 if family == FAMILIES[0] else 0.0) for family in FAMILIES}
    model = AFTResidualDelayModel(
        aft_models=aft_models,
        weights=weights,
        features=["x"],
        calibration=_calibration(0.0),
        fallback_model=_ConstantModel(999.0),
    )
    frame = pd.DataFrame(
        {
            "x": [1.0],
            "snapshot_date": ["2020-01-01"],
            "planned_completion_date": ["2020-01-05"],
            "lifecycle_stage": ["mid"],
        }
    )
    # Snapshot + 10 predicted remaining days = Jan 11; planned Jan 5 => 6 days delay.
    np.testing.assert_allclose(model.predict(frame), [6.0])


def test_production_baseline_names_identify_combined_promotion():
    assert "exp33" in PRODUCTION_COST_BASELINE
    assert "exp32" in PRODUCTION_DELAY_BASELINE
    assert "exp33" in PRODUCTION_DELAY_BASELINE
    assert "exp34_fallback" in PRODUCTION_DELAY_BASELINE
