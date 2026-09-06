import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp135_log_space_cost import (
    Exp135CostProductionModel,
    inv_signed_log,
    signed_log,
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
        return np.full(len(frame), 0.2)


def test_signed_log_round_trip_supports_negative_and_positive_values():
    values = np.array([-100.0, -5.0, 0.0, 5.0, 100.0])
    assert np.allclose(inv_signed_log(signed_log(values)), values)


def test_exp135_wrapper_reconstructs_prediction_in_original_space():
    model = Exp135CostProductionModel(
        base_model=_CostBase(),
        factor_scaler=_IdentityScaler(),
        factor_model=_TwoFactors(),
        factor_features=["cost_escalation_percentage", "duration_ratio"],
        factor_medians={"cost_escalation_percentage": 0.0, "duration_ratio": 1.0},
        booster=_CostBooster(),
        booster_features=["production_prediction", "exp105_factor_1", "exp105_factor_2"],
        booster_medians={"production_prediction": 20.0, "exp105_factor_1": 0.0, "exp105_factor_2": 1.0},
        correction_cap=0.1,
        correction_scale=0.5,
        input_features=["cost_escalation_percentage", "duration_ratio"],
    )
    frame = pd.DataFrame(
        {"cost_escalation_percentage": [2.0, 3.0], "duration_ratio": [1.1, 1.2]}
    )
    expected = inv_signed_log(signed_log(np.array([20.0, 20.0])) + 0.05)
    assert np.allclose(model.predict(frame), expected)
