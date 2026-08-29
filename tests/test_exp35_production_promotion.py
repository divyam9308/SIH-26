import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import FAMILIES
from backend.app.ml.production_exp35_baseline import (
    AFTResidualDelayModel,
    CALIBRATION_GATE_FEATURE,
    ResidualCalibratedCostModel,
    PRODUCTION_COST_BASELINE,
    PRODUCTION_DELAY_BASELINE,
    VERIFIED_AFT_CALIBRATION_PROJECTS,
    _select_aft_calibration_projects,
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


def _delay_model(fallback=999.0):
    aft_models = {family: _ConstantModel(np.log1p(10.0)) for family in FAMILIES}
    weights = {family: (1.0 if family == FAMILIES[0] else 0.0) for family in FAMILIES}
    return AFTResidualDelayModel(
        aft_models=aft_models,
        weights=weights,
        features=["x"],
        calibration=_calibration(0.0),
        fallback_model=_ConstantModel(fallback),
    )


def test_cost_wrapper_applies_exp33_residual_calibration():
    model = ResidualCalibratedCostModel(
        _ConstantModel(10.0), ["x"], _calibration(2.5)
    )
    frame = pd.DataFrame({"x": [1.0, 2.0], "lifecycle_stage": ["early", "late"]})
    np.testing.assert_allclose(model.predict(frame), [12.5, 12.5])


def test_delay_wrapper_preserves_exp34_fallback_without_planned_completion():
    model = _delay_model(fallback=123.0)
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
    model = _delay_model()
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


def test_delay_wrapper_uses_exp34_outside_fixed_calibration_gate():
    model = _delay_model(fallback=321.0)
    frame = pd.DataFrame(
        {
            "x": [1.0],
            "snapshot_date": ["2020-01-01"],
            "planned_completion_date": ["2020-01-05"],
            "lifecycle_stage": ["mid"],
            CALIBRATION_GATE_FEATURE: [False],
        }
    )
    np.testing.assert_allclose(model.predict(frame), [321.0])


def test_missing_historical_gate_does_not_disable_live_aft():
    model = _delay_model()
    frame = pd.DataFrame(
        {
            "x": [1.0],
            "snapshot_date": ["2020-01-01"],
            "planned_completion_date": ["2020-01-05"],
            "lifecycle_stage": ["mid"],
            CALIBRATION_GATE_FEATURE: [np.nan],
        }
    )
    np.testing.assert_allclose(model.predict(frame), [6.0])


def test_aft_calibration_cohort_selection_is_fixed_and_evidence_only():
    rows = []
    for project, eligible, total in [
        ("A", 3, 3),
        ("B", 2, 2),
        ("C", 2, 3),
        ("D", 1, 3),
    ]:
        for i in range(total):
            rows.append(
                {
                    "canonical_project_id": project,
                    "snapshot_date": f"2020-01-{i + 1:02d}",
                    "planned_completion_date": "2020-02-01" if i < eligible else None,
                    # These deliberately extreme target-like columns must not affect selection.
                    "actual_delay_days": 10000 if project == "D" else 0,
                }
            )
    frame = pd.DataFrame(rows)
    selected = _select_aft_calibration_projects(frame, limit=2)
    assert selected == {"A", "B"}


def test_production_baseline_names_identify_688_project_combined_promotion():
    assert VERIFIED_AFT_CALIBRATION_PROJECTS == 688
    assert "exp33" in PRODUCTION_COST_BASELINE
    assert "exp32" in PRODUCTION_DELAY_BASELINE
    assert "exp33" in PRODUCTION_DELAY_BASELINE
    assert "688" in PRODUCTION_DELAY_BASELINE
    assert "exp34_fallback" in PRODUCTION_DELAY_BASELINE
