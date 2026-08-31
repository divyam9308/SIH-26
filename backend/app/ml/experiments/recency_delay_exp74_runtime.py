"""Runtime OOF fix for Experiment 74.

Exp74 needs a deeper forward OOF history than the production U1 trainer itself.
The production helper intentionally caps rolling folds at three because that is
sufficient to fit the deployed U1 residual booster. Reusing that capped helper
inside Exp74 leaves at most two *current-production* OOF folds after the first
fold is reserved to fit U1, which makes Exp74's recency-policy selection
impossible.

This module keeps production untouched and expands folds only inside the isolated
Exp74 challenger. Importing it installs the corrected OOF builder into the Exp74
module before ``fit_experiment`` or ``main`` runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments import recency_delay_exp74 as exp74
import backend.app.ml.production_u1_delay_baseline as u1_production

# Enough base folds to support up to five selection folds after one fold is
# consumed to establish the U1 residual layer, while remaining bounded for CI.
EXPANDED_BASE_OOF_FOLDS = exp74.SELECTION_FOLDS + 3


def _forward_base_folds(train_delay: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Build deeper Exp74-only forward folds using the production fold criteria."""
    years = sorted(
        int(y)
        for y in pd.to_numeric(
            train_delay["completion_year"], errors="coerce"
        ).dropna().unique()
    )
    completion_year = pd.to_numeric(train_delay["completion_year"], errors="coerce")
    folds: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    for year in reversed(years[1:]):
        fitting = train_delay.loc[completion_year.lt(year)].copy()
        validation = train_delay.loc[completion_year.eq(year)].copy()
        if (
            fitting["canonical_project_id"].nunique() >= 10
            and validation["canonical_project_id"].nunique() >= 3
        ):
            folds.append((fitting, validation, int(year)))
        if len(folds) >= EXPANDED_BASE_OOF_FOLDS:
            break
    return list(reversed(folds))


def _expanded_base_oof(prior_train: pd.DataFrame, base_delay_model) -> pd.DataFrame:
    """Reproduce the Exp61 base Delay OOF path with more forward folds for Exp74."""
    features = list(base_delay_model.model_features)
    train_delay = u1_production._remaining_frame(prior_train)
    folds = _forward_base_folds(train_delay)
    if len(folds) < 4:
        raise ValueError(
            "Exp74 requires at least four eligible base Delay folds to form "
            "three current-production forward OOF folds"
        )

    chunks = []
    for fit, val, year in folds:
        models = u1_production._fit_aft_family_models(fit, features)
        remaining = u1_production._aft_remaining_prediction(
            models,
            base_delay_model.weights,
            val,
            features,
        )
        raw = u1_production._delay_from_remaining(val, remaining)
        prediction = np.maximum(
            0.0,
            raw + u1_production._corrections(
                val,
                raw,
                base_delay_model.calibration,
            ),
        )
        part = val.copy()
        part["production_prediction"] = prediction
        part["residual"] = (
            pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float)
            - prediction
        )
        part["oof_year"] = int(year)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def _current_production_oof(
    prior_train: pd.DataFrame,
    production_delay_model,
) -> pd.DataFrame:
    """Build forward OOF errors for the full post-PR110 production Delay model."""
    if not hasattr(production_delay_model, "base_model"):
        raise TypeError("Exp74 requires the post-PR110 U1 Delay production wrapper")

    # Important: do not call u1_production._delay_oof_frame here. That helper is
    # deliberately capped at three folds for production training and caused the
    # original Exp74 CI failure.
    base_oof = _expanded_base_oof(
        prior_train,
        production_delay_model.base_model,
    )
    fold_year = pd.to_numeric(base_oof["oof_year"], errors="coerce")
    years = sorted(int(x) for x in fold_year.dropna().unique())
    chunks = []
    for year in years[1:]:
        fit = base_oof.loc[fold_year < year].copy()
        val = base_oof.loc[fold_year == year].copy()
        if len(fit) < exp74.MIN_POLICY_ROWS or val.empty:
            continue
        _, _, _, _, correction = u1_production._fit_u1_booster(fit, val)
        anchor = pd.to_numeric(
            val["production_prediction"], errors="coerce"
        ).to_numpy(float)
        prediction = np.maximum(0.0, anchor + correction)
        part = val.copy()
        part["production_prediction"] = prediction
        part["residual"] = (
            pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float)
            - prediction
        )
        part["oof_year"] = int(year)
        chunks.append(part)

    if len(chunks) < 3:
        raise ValueError(
            "Exp74 could not form three current-production forward OOF folds "
            "after expanding the experiment-only base history"
        )
    return pd.concat(chunks, ignore_index=True)


# ``fit_experiment`` resolves this name from the Exp74 module at call time, so
# installing the corrected builder here fixes both CLI and adapter entrypoints
# without modifying production code.
exp74._current_production_oof = _current_production_oof

EXPERIMENT_ID = exp74.EXPERIMENT_ID
EXPERIMENT_NAME = exp74.EXPERIMENT_NAME
EXPERIMENT_SCOPE = exp74.EXPERIMENT_SCOPE
EXPERIMENT_SEQUENCE = exp74.EXPERIMENT_SEQUENCE


def fit_experiment(**kwargs):
    return exp74.fit_experiment(**kwargs)


def main() -> None:
    exp74.main()


if __name__ == "__main__":
    main()
