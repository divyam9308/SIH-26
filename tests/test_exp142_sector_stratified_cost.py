import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp142_sector_stratified_cost import (
    SECTOR_COST_RESIDUAL_CAPS,
    _normalize_sector_name,
    add_cost_structural_features,
    apply_sector_cost_caps,
    sector_cost_cap_arrays,
)


def test_normalize_sector_name_mapping():
    assert _normalize_sector_name("Northern Railway Project") == "railways"
    assert _normalize_sector_name("Urban Development Metro") == "urban development"
    assert _normalize_sector_name("National Highway Transport") == "road transport and highways"
    assert _normalize_sector_name("Hydro Power Station") == "power"
    assert _normalize_sector_name("Petroleum Pipeline") == "petroleum and natural gas"
    assert _normalize_sector_name("Telecom Tower Project") == "telecommunications"
    assert _normalize_sector_name("Coal Mine Block") == "coal"
    assert _normalize_sector_name("Water Resources Canal") == "water resources"
    assert _normalize_sector_name("Unknown Sector") == "default"


def test_sector_cost_cap_arrays_and_clipping():
    frame = pd.DataFrame({"sector": ["Railways", "Telecommunications", "Unknown"]})
    lower, upper, names = sector_cost_cap_arrays(frame)

    assert names == ["railways", "telecommunications", "default"]
    assert lower[0] == -15.0 and upper[0] == 65.0
    assert lower[1] == -5.0 and upper[1] == 12.0
    assert lower[2] == -10.0 and upper[2] == 35.0

    raw = np.array([100.0, 50.0, -30.0])
    clipped = apply_sector_cost_caps(frame, raw)
    assert np.allclose(clipped, [65.0, 12.0, -10.0])


def test_add_cost_structural_features():
    frame = pd.DataFrame(
        {
            "sector": ["Railways", "Power"],
            "approved_cost_cr": [1500.0, 500.0],
            "expenditure_ratio": [0.1, 0.5],
            "duration_ratio": [1.5, 1.0],
        }
    )
    out = add_cost_structural_features(frame)
    assert out["is_railways_sector"].tolist() == [1.0, 0.0]
    assert out["is_high_cost_mega_project"].tolist() == [1.0, 0.0]
    assert out["stagnant_spend_lag"].tolist() == [1.0, 0.0]
