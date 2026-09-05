from __future__ import annotations

from functools import lru_cache
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import PROCESSED_DIR, RAW_DIR
from backend.app.ml.features import engineer_temporal_features, load_project_history

DATE_COLUMNS = ["snapshot_date", "original_end_date", "revised_end_date"]
PROJECT_DATASET = PROCESSED_DIR / "model_dataset.csv"


def project_dataset_signature() -> str:
    """Return a content key that changes whenever the processed dataset changes."""
    stat = PROJECT_DATASET.stat()
    digest = hashlib.sha256(PROJECT_DATASET.read_bytes()).hexdigest()
    return f"{stat.st_size}:{digest}"


@lru_cache(maxsize=1)
def _projects_df_cached(dataset_signature: str) -> pd.DataFrame:
    del dataset_signature
    df = pd.read_csv(PROJECT_DATASET, dtype={"project_code": str})
    for c in DATE_COLUMNS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def projects_df() -> pd.DataFrame:
    return _projects_df_cached(project_dataset_signature())


@lru_cache(maxsize=1)
def history_df() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "paimana_high_value_history.csv", dtype={"project_code": str})
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["revised_completion_date"] = pd.to_datetime(df["revised_completion_date"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def temporal_features_df() -> pd.DataFrame:
    return engineer_temporal_features(load_project_history())


def latest_temporal_snapshot(code: str) -> pd.Series:
    df = temporal_features_df()
    hit = df[df["project_id"].astype(str) == str(code)].sort_values("month")
    if hit.empty:
        raise KeyError(code)
    return hit.iloc[-1]


def _json_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {k: _json_value(v) for k, v in row.items()}


def get_project(code: str) -> pd.Series:
    hit = projects_df()[projects_df()["project_code"].astype(str) == str(code)]
    if hit.empty:
        raise KeyError(code)
    return hit.iloc[0]


def list_projects(search: str | None = None, sector: str | None = None) -> pd.DataFrame:
    df = projects_df().copy()
    if search:
        q = search.lower().strip()
        mask = (
            df["project_name"].str.lower().str.contains(q, na=False)
            | df["project_code"].astype(str).str.contains(q, na=False)
            | df["ministry"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]
    if sector:
        df = df[df["sector"] == sector]
    return df


def sectors() -> list[str]:
    return sorted(projects_df()["sector"].dropna().unique().tolist())
