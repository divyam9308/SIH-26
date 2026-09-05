"""Execution-only helpers for strict forward Cost production OOF sharding.

This module does not change the experiment or model contract. It only lets the
same strict production OOF folds be generated independently, serialized as
workflow artifacts, and recombined before challenger selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors

TRAINING_START = 2001
TRAINING_END = 2021
TEST_END = 2025
OOF_YEARS = (2018, 2019, 2020, 2021)
_REQUIRED_COLUMNS = {
    "canonical_project_id",
    "actual_cost_overrun_percentage",
    "sample_weight",
    "production_prediction",
    "production_residual",
    "oof_year",
}


def build_oof_training_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build only the frozen training frame needed to score one strict OOF fold."""
    data, identity = build_training_dataset()
    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(
        prepared,
        TRAINING_START,
        TRAINING_END,
        TEST_END,
    )
    train, _, _ = _build_temporal_delay_priors(train, test)
    return data, identity, train


def expected_oof_years(
    train: pd.DataFrame,
    forward_folds: Callable,
    *,
    max_folds: int = 4,
) -> tuple[int, ...]:
    years = tuple(
        int(year)
        for _, _, year in forward_folds(train, max_folds)
        if int(year) - 1 >= 2005
    )
    if years != OOF_YEARS:
        raise RuntimeError(
            f"Strict Cost OOF contract changed: expected {OOF_YEARS}, got {years}"
        )
    return years


def _validate_part(part: pd.DataFrame, year: int) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(part.columns))
    if missing:
        raise ValueError(f"OOF shard {year} is missing required columns: {missing}")
    years = set(pd.to_numeric(part["oof_year"], errors="coerce").dropna().astype(int))
    if years != {int(year)}:
        raise ValueError(f"OOF shard {year} contains unexpected years: {sorted(years)}")
    if part.empty:
        raise ValueError(f"OOF shard {year} is empty")
    prediction = pd.to_numeric(part["production_prediction"], errors="coerce").to_numpy(float)
    residual = pd.to_numeric(part["production_residual"], errors="coerce").to_numpy(float)
    if not np.isfinite(prediction).all() or not np.isfinite(residual).all():
        raise ValueError(f"OOF shard {year} contains non-finite production evidence")


def generate_oof_shard(
    *,
    year: int,
    forward_folds: Callable,
    fold_trainer: Callable,
    output_path: Path,
) -> dict:
    """Generate exactly one unchanged strict-production OOF fold."""
    year = int(year)
    data, identity, train = build_oof_training_context()
    expected_oof_years(train, forward_folds, max_folds=4)
    validation_by_year = {
        int(fold_year): validation.copy()
        for _, validation, fold_year in forward_folds(train, 4)
        if int(fold_year) - 1 >= 2005
    }
    if year not in validation_by_year:
        raise ValueError(f"OOF year {year} is outside the frozen fold contract {OOF_YEARS}")

    part = fold_trainer(validation_by_year[year], year, data, identity)
    _validate_part(part, year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part.to_pickle(output_path)
    return {
        "year": year,
        "rows": int(len(part)),
        "projects": int(part["canonical_project_id"].nunique()),
        "path": str(output_path),
    }


def load_oof_shards(
    oof_dir: Path,
    *,
    expected_years: tuple[int, ...] = OOF_YEARS,
) -> pd.DataFrame:
    """Load and validate one artifact per frozen OOF year, then concatenate."""
    parts: list[pd.DataFrame] = []
    for year in expected_years:
        path = oof_dir / f"production_oof_{int(year)}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing strict production OOF shard: {path}")
        part = pd.read_pickle(path)
        _validate_part(part, int(year))
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    combined_years = tuple(
        sorted(
            int(value)
            for value in pd.to_numeric(combined["oof_year"], errors="coerce")
            .dropna()
            .unique()
        )
    )
    if combined_years != tuple(expected_years):
        raise ValueError(
            f"Combined OOF years changed: expected {expected_years}, got {combined_years}"
        )
    return combined


def validate_oof_against_context(
    production_oof: pd.DataFrame,
    train: pd.DataFrame,
    forward_folds: Callable,
    *,
    max_folds: int = 4,
) -> pd.DataFrame:
    """Ensure downloaded shards are exactly the folds the live experiment expects."""
    expected = expected_oof_years(train, forward_folds, max_folds=max_folds)
    actual = tuple(
        sorted(
            int(value)
            for value in pd.to_numeric(production_oof["oof_year"], errors="coerce")
            .dropna()
            .unique()
        )
    )
    if actual != expected:
        raise ValueError(f"Precomputed OOF fold mismatch: expected {expected}, got {actual}")
    return production_oof.copy()
