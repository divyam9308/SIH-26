import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp139_sector_stratified_delay import (
    SECTOR_RESIDUAL_CAPS,
    add_structural_lag_features,
    apply_sector_caps,
    sector_cap_arrays,
)


def test_sector_cap_arrays_map_known_and_default_sectors():
    frame = pd.DataFrame(
        {
            "sector": [
                "Railways",
                "Urban Development",
                "Road Transport & Highways",
                "Power",
                "Petroleum and Natural Gas",
                "Telecommunications",
                "Coal",
                "Water Resources",
                "Ports",
            ]
        }
    )
    lower, upper, names = sector_cap_arrays(frame)
    assert names == [
        "railways",
        "urban development",
        "road transport and highways",
        "power",
        "petroleum and natural gas",
        "telecommunications",
        "coal",
        "water resources",
        "default",
    ]
    for idx, name in enumerate(names):
        assert lower[idx] == SECTOR_RESIDUAL_CAPS[name][0]
        assert upper[idx] == SECTOR_RESIDUAL_CAPS[name][1]


def test_apply_sector_caps_uses_row_specific_bounds():
    frame = pd.DataFrame({"sector": ["Railways", "Telecommunications", "Ports"]})
    clipped = apply_sector_caps(frame, np.asarray([4000.0, 900.0, -900.0]))
    assert np.allclose(clipped, [2500.0, 400.0, -450.0])


def test_structural_lag_features_match_specification():
    frame = pd.DataFrame(
        {
            "sector": ["Railways", "Power"],
            "elapsed_duration_days": [4000.0, 1000.0],
            "cost_growth_velocity_6m": [-0.1, 0.2],
            "duration_ratio": [1.8, 1.2],
        }
    )
    out = add_structural_lag_features(frame)
    assert out["is_railways_sector"].tolist() == [1.0, 0.0]
    assert out["elapsed_over_10yr"].tolist() == [1.0, 0.0]
    assert out["stagnant_progress_24m"].tolist() == [1.0, 0.0]
