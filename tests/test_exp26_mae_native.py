from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.mae_native_exp26 import (
    COMPARISON_FILTER,
    EXPERIMENT_ID,
    _exp12_comparable,
    _mae_regressors,
    _overall_verdict,
    _target_verdict,
)
from backend.app.ml.experiments.trajectory_exp12 import MIN_HISTORY
from backend.app.ml.monthly_training import _fit_pipeline, _regressors


def test_exp26_adapter_contract_is_discoverable():
    adapter = get_experiment_adapter(EXPERIMENT_ID)
    assert adapter.sequence == 26
    assert adapter.scope == "cost+delay"
    assert callable(adapter.module.fit_against_production)
    assert callable(adapter.module.filter_comparable_rows)
    assert callable(adapter.module.predict_project)


def test_mae_regressors_change_only_loss_and_keep_production_shape():
    seed = 26203
    production = _regressors(seed)
    candidate = _mae_regressors(seed)
    assert set(candidate) == set(production) == {"lightgbm", "xgboost", "extra_trees"}

    l2_lgb = production["lightgbm"].get_params()
    l1_lgb = candidate["lightgbm"].get_params()
    for name in ("n_estimators", "learning_rate", "max_depth", "num_leaves", "random_state"):
        assert l1_lgb[name] == l2_lgb[name]
    assert l1_lgb["objective"] == "regression_l1"

    l2_xgb = production["xgboost"].get_params()
    l1_xgb = candidate["xgboost"].get_params()
    for name in (
        "n_estimators", "learning_rate", "max_depth", "subsample",
        "colsample_bytree", "random_state", "n_jobs",
    ):
        assert l1_xgb[name] == l2_xgb[name]
    assert l2_xgb["objective"] == "reg:squarederror"
    assert l1_xgb["objective"] == "reg:absoluteerror"

    l2_et = production["extra_trees"].get_params()
    l1_et = candidate["extra_trees"].get_params()
    for name in ("n_estimators", "min_samples_leaf", "max_features", "random_state", "n_jobs"):
        assert l1_et[name] == l2_et[name]
    assert l2_et["criterion"] == "squared_error"
    assert l1_et["criterion"] == "absolute_error"


def test_all_mae_native_families_fit_through_production_pipeline():
    frame = pd.DataFrame({
        "numeric": np.linspace(0.0, 11.0, 12),
        "category": ["a", "b", "c"] * 4,
        "target": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 80.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        "sample_weight": np.ones(12),
    })
    for model in _mae_regressors(123).values():
        fitted = _fit_pipeline(model, frame, ["numeric", "category"], "target")
        pred = fitted.predict(frame[["numeric", "category"]])
        assert len(pred) == len(frame)
        assert np.isfinite(pred).all()


def test_exp26_uses_exact_exp12_comparable_filter_and_reweights_after_filtering():
    frame = pd.DataFrame({
        "canonical_project_id": ["A", "A", "A", "B", "B", "C"],
        "exp12_history_12m": [MIN_HISTORY - 1, MIN_HISTORY, MIN_HISTORY + 1, MIN_HISTORY, MIN_HISTORY + 3, 0],
        "sample_weight": [99.0] * 6,
    })
    compare = _exp12_comparable(frame)
    assert set(compare.canonical_project_id) == {"A", "B"}
    assert len(compare) == 4
    assert compare.exp12_history_12m.ge(MIN_HISTORY).all()
    totals = compare.groupby("canonical_project_id").sample_weight.sum()
    assert np.allclose(totals.to_numpy(dtype=float), 1.0)
    assert str(MIN_HISTORY) in COMPARISON_FILTER
    assert "Exp12" in COMPARISON_FILTER


def test_target_and_pr_level_verdicts_are_explicit():
    assert _target_verdict(0.1) == "PROMOTION CANDIDATE"
    assert _target_verdict(0.0) == "NO CHANGE"
    assert _target_verdict(-0.1) == "REGRESSION / DO NOT PROMOTE"
    assert _overall_verdict(0.2, 0.1) == "PROMOTION CANDIDATE"
    assert _overall_verdict(0.2, -0.01) == "REGRESSION / DO NOT PROMOTE"
