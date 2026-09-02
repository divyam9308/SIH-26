from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.experiments.experiment_e_delay_weather_exposure import (
    WEATHER_FEATURES,
    attach_weather,
    load_verified_weather,
)


def test_missing_verified_weather_returns_none(tmp_path: Path):
    assert load_verified_weather(tmp_path / "missing.csv") is None


def test_weather_rows_require_source_urls(tmp_path: Path):
    path = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "month": ["2020-01-01"],
            "state": ["Punjab"],
            "rainfall_anomaly_pct": [25],
            "extreme_rain_days": [2],
            "flood_event": [0],
            "cyclone_event": [0],
            "heatwave_days": [0],
            "source_url": [""],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_verified_weather(path)


def test_weather_attachment_uses_snapshot_month_only(tmp_path: Path):
    path = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "month": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "state": ["Punjab"] * 3,
            "rainfall_anomaly_pct": [10, 20, 30],
            "extreme_rain_days": [1, 2, 3],
            "flood_event": [0, 1, 0],
            "cyclone_event": [0, 0, 0],
            "heatwave_days": [0, 0, 1],
            "source_url": ["https://example.gov.in/a"] * 3,
        }
    ).to_csv(path, index=False)
    weather = load_verified_weather(path)
    score = pd.DataFrame({"state": ["Punjab"], "snapshot_date": ["2020-02-15"]})
    out = attach_weather(score, weather)
    for feature in WEATHER_FEATURES:
        assert feature in out
    assert out.loc[0, "exp_e_extreme_rain_days_3m"] == 3
