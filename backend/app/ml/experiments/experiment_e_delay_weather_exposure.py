"""Experiment E: verified historical weather/disaster exposure for Delay.

Unlike the earlier calendar-monsoon proxy, this experiment accepts only a
tracked, source-attributed monthly state weather file.  If that evidence is not
present, the challenger is exactly production and reports the data gap rather
than fabricating rainfall/flood/cyclone values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from backend.app.ml.experiments.post_exp113_delay_common import (
    fit_residual,
    persist,
    prepare_context,
    production_oof,
)

EXPERIMENT_ID = "exp_e"
EXPERIMENT_NAME = "E — verified weather and disaster exposure"
SEED = 13401
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WEATHER_PATH = ROOT / "data" / "external" / "imd_state_weather_monthly.csv"
REQUIRED = [
    "month",
    "state",
    "rainfall_anomaly_pct",
    "extreme_rain_days",
    "flood_event",
    "cyclone_event",
    "heatwave_days",
    "source_url",
]
WEATHER_FEATURES = [
    "exp_e_rainfall_anomaly_pct",
    "exp_e_extreme_rain_days_3m",
    "exp_e_flood_events_6m",
    "exp_e_cyclone_events_12m",
    "exp_e_heatwave_days_3m",
    "exp_e_weather_shock_index",
]
RESIDUAL_FEATURES = [
    "production_prediction",
    "duration_ratio",
    "schedule_slippage_days",
    "physical_progress",
    "expenditure_ratio",
    "progress_deviation",
    "exp58_delay_hier_prior",
    *WEATHER_FEATURES,
]


def _normalise_state(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna("<NA>")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def load_verified_weather(path: Path = DEFAULT_WEATHER_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    weather = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in weather.columns]
    if missing:
        raise ValueError(f"Verified weather file missing columns: {missing}")
    if weather["source_url"].isna().any() or not weather["source_url"].astype(str).str.startswith("https://").all():
        raise ValueError("Every weather row must carry a source-attributed https URL")
    weather["month"] = pd.to_datetime(weather["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if weather["month"].isna().any():
        raise ValueError("Verified weather file contains invalid month values")
    weather["_state"] = _normalise_state(weather["state"])
    for col in ["rainfall_anomaly_pct", "extreme_rain_days", "flood_event", "cyclone_event", "heatwave_days"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce").fillna(0.0)
    weather = weather.sort_values(["_state", "month"], kind="mergesort").copy()
    g = weather.groupby("_state", sort=False)
    weather["_rain3"] = g["extreme_rain_days"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    weather["_flood6"] = g["flood_event"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    weather["_cyclone12"] = g["cyclone_event"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    weather["_heat3"] = g["heatwave_days"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    return weather


def attach_weather(score: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    out = score.copy()
    out["_state"] = _normalise_state(out.get("state", pd.Series("<NA>", index=out.index)))
    out["_month"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    keep = weather[["_state", "month", "rainfall_anomaly_pct", "_rain3", "_flood6", "_cyclone12", "_heat3"]].rename(columns={"month": "_month"})
    out = out.merge(keep, on=["_state", "_month"], how="left", sort=False)
    out["exp_e_rainfall_anomaly_pct"] = pd.to_numeric(out["rainfall_anomaly_pct"], errors="coerce")
    out["exp_e_extreme_rain_days_3m"] = pd.to_numeric(out["_rain3"], errors="coerce")
    out["exp_e_flood_events_6m"] = pd.to_numeric(out["_flood6"], errors="coerce")
    out["exp_e_cyclone_events_12m"] = pd.to_numeric(out["_cyclone12"], errors="coerce")
    out["exp_e_heatwave_days_3m"] = pd.to_numeric(out["_heat3"], errors="coerce")
    out["exp_e_weather_shock_index"] = (
        out["exp_e_rainfall_anomaly_pct"].clip(lower=0).fillna(0) / 100.0
        + out["exp_e_extreme_rain_days_3m"].fillna(0) / 10.0
        + out["exp_e_flood_events_6m"].fillna(0)
        + out["exp_e_cyclone_events_12m"].fillna(0)
        + out["exp_e_heatwave_days_3m"].fillna(0) / 20.0
    )
    return out.drop(columns=["_state", "_month", "rainfall_anomaly_pct", "_rain3", "_flood6", "_cyclone12", "_heat3"], errors="ignore")


def fit_experiment(training_end: int, output: str, weather_path: Path = DEFAULT_WEATHER_PATH):
    ctx = prepare_context(training_end)
    weather = load_verified_weather(weather_path)
    if weather is None:
        details = {
            "changed_dimension": "verified_weather_exposure",
            "verified_external_data_available": False,
            "required_path": str(weather_path.relative_to(ROOT) if weather_path.is_relative_to(ROOT) else weather_path),
            "proxy_substitution_allowed": False,
            "reason": "No verified source-attributed weather dataset is tracked; production retained exactly.",
        }
        return persist(EXPERIMENT_ID, EXPERIMENT_NAME, ctx, ctx["production_delay"].copy(), details, output)

    oof = production_oof(ctx, max_folds=6)
    meta = attach_weather(oof, weather)
    score = ctx["cohort"].copy()
    score["production_prediction"] = ctx["production_delay"]
    score = attach_weather(score, weather)
    correction, details = fit_residual(meta, score, RESIDUAL_FEATURES, SEED)
    details.update(
        {
            "changed_dimension": "verified_weather_exposure",
            "verified_external_data_available": True,
            "weather_rows": len(weather),
            "source_urls": int(weather["source_url"].nunique()),
            "weather_features": WEATHER_FEATURES,
            "holdout_outcomes_used_for_weather": False,
        }
    )
    return persist(EXPERIMENT_ID, EXPERIMENT_NAME, ctx, ctx["production_delay"] + correction, details, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=int, choices=[2021, 2022], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weather-path", type=Path, default=DEFAULT_WEATHER_PATH)
    a = p.parse_args()
    fit_experiment(a.end, a.output, a.weather_path)


if __name__ == "__main__":
    main()
