import numpy as np
import pandas as pd

from backend.app.ml.production_exp105_exp113_baseline import (
    Exp105CostProductionModel,
    Exp113DelayProductionModel,
)


class _CostBase:
    def predict(self, frame):
        return np.full(len(frame), 20.0)


class _IdentityScaler:
    def transform(self, frame):
        return np.asarray(frame, dtype=float)


class _TwoFactors:
    def transform(self, frame):
        x = np.asarray(frame, dtype=float)
        return x[:, :2]


class _CostBooster:
    def predict(self, frame):
        return np.full(len(frame), 100.0)


def test_exp105_wrapper_applies_scaled_bounded_residual():
    model = Exp105CostProductionModel(
        base_model=_CostBase(),
        factor_scaler=_IdentityScaler(),
        factor_model=_TwoFactors(),
        factor_features=["cost_escalation_percentage", "duration_ratio"],
        factor_medians={"cost_escalation_percentage": 0.0, "duration_ratio": 1.0},
        booster=_CostBooster(),
        booster_features=["production_prediction", "exp105_factor_1", "exp105_factor_2"],
        booster_medians={"production_prediction": 20.0, "exp105_factor_1": 0.0, "exp105_factor_2": 1.0},
        correction_cap=10.0,
        correction_scale=0.5,
        input_features=["cost_escalation_percentage", "duration_ratio"],
    )
    frame = pd.DataFrame({"cost_escalation_percentage": [2.0, 3.0], "duration_ratio": [1.1, 1.2]})
    assert np.allclose(model.predict(frame), [25.0, 25.0])


class _Exp61Base:
    def predict(self, frame):
        return np.full(len(frame), 80.0)


class _U1Base:
    booster_prior_state = None
    base_model = _Exp61Base()

    def predict(self, frame):
        return np.full(len(frame), 100.0)


class _QuantileModel:
    def __init__(self, remaining):
        self.value = float(np.log1p(remaining))

    def predict(self, frame):
        return np.full(len(frame), self.value)


class _DelayBooster:
    def predict(self, frame):
        return np.full(len(frame), -100.0)


def test_exp113_wrapper_keeps_u1_anchor_and_applies_bounded_correction():
    model = Exp113DelayProductionModel(
        base_model=_U1Base(),
        quantile_models={
            0.25: _QuantileModel(100.0),
            0.5: _QuantileModel(200.0),
            0.75: _QuantileModel(300.0),
        },
        quantile_features=["production_prediction"],
        quantile_medians={"production_prediction": 100.0},
        booster=_DelayBooster(),
        booster_features=["production_prediction", "u1_correction", "exp113_interval_width"],
        booster_medians={"production_prediction": 100.0, "u1_correction": 20.0, "exp113_interval_width": 200.0},
        correction_cap=20.0,
        correction_scale=0.5,
        input_features=["snapshot_date", "planned_completion_date"],
    )
    frame = pd.DataFrame(
        {
            "snapshot_date": ["2024-01-01", "2024-02-01"],
            "planned_completion_date": ["2024-12-31", "2024-12-31"],
        }
    )
    assert np.allclose(model.predict(frame), [90.0, 90.0])
