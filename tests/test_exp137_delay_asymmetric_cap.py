import pandas as pd

from backend.app.ml.experiments.exp137_delay_asymmetric_cap import (
    LOWER_CAP_DAYS,
    UPPER_CAP_DAYS,
    add_structural_lag_features,
)


def test_exp137_structural_lag_features():
    frame = pd.DataFrame(
        {
            "sector": ["Railways", "Power"],
            "elapsed_duration_days": [4000, 1200],
            "cost_growth_velocity_6m": [-0.1, 0.2],
            "duration_ratio": [1.8, 1.1],
        }
    )
    out = add_structural_lag_features(frame)
    assert out["is_railways_sector"].tolist() == [1.0, 0.0]
    assert out["elapsed_over_10yr"].tolist() == [1.0, 0.0]
    assert out["stagnant_progress_24m"].tolist() == [1.0, 0.0]


def test_exp137_caps_are_asymmetric():
    assert LOWER_CAP_DAYS == -500.0
    assert UPPER_CAP_DAYS == 1500.0
