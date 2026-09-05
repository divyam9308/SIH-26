import numpy as np
import pandas as pd
import pytest

from backend.app.ml.experiments.exp94_family_disagreement_cost import (
    FAMILIES,
    META_FEATURES,
    SCALE_GRID,
    _add_disagreement_columns,
    select_scale,
    weighted_metrics,
    window_contract,
)


def test_window_contract_is_limited_to_original_exp94_windows():
    assert window_contract(2019) == (2020, 2025)
    assert window_contract(2021) == (2022, 2025)
    with pytest.raises(ValueError):
        window_contract(2022)


def test_family_disagreement_is_computed_from_all_three_family_predictions():
    frame = pd.DataFrame(
        {
            "exp94_extra_trees": [10.0, 20.0],
            "exp94_lightgbm": [20.0, 20.0],
            "exp94_xgboost": [30.0, 20.0],
        }
    )
    result = _add_disagreement_columns(frame)
    assert tuple(FAMILIES) == ("extra_trees", "lightgbm", "xgboost")
    assert result.loc[0, "exp94_family_mean"] == pytest.approx(20.0)
    assert result.loc[0, "exp94_family_range"] == pytest.approx(20.0)
    assert result.loc[0, "exp94_family_std"] > 0
    assert result.loc[1, "exp94_family_std"] == pytest.approx(0.0)
    assert result.loc[1, "exp94_family_range"] == pytest.approx(0.0)


def _meta_frame(actual, production, correction):
    n = len(actual)
    return pd.DataFrame(
        {
            "actual_cost_overrun_percentage": np.asarray(actual, dtype=float),
            "production_prediction": np.asarray(production, dtype=float),
            "exp94_raw_correction": np.asarray(correction, dtype=float),
            "sample_weight": np.ones(n, dtype=float),
            "canonical_project_id": [f"p{i}" for i in range(n)],
            "meta_validation_year": [2018] * (n // 2) + [2019] * (n - n // 2),
        }
    )


def test_scale_selection_can_choose_positive_correction_when_mae_improves():
    frame = _meta_frame(
        actual=[0.0, 10.0, 20.0, 100.0],
        production=[0.0, 10.0, 20.0, 60.0],
        correction=[0.0, 0.0, 0.0, 40.0],
    )
    result = select_scale(frame)
    assert result["selected"]["scale"] in SCALE_GRID
    assert result["selected"]["scale"] > 0.0
    assert result["selected"]["MAE"] < result["baseline"]["MAE"]


def test_scale_selection_falls_back_to_production_when_correction_hurts():
    frame = _meta_frame(
        actual=[0.0, 10.0, 20.0, 100.0],
        production=[0.0, 10.0, 20.0, 60.0],
        correction=[50.0, 50.0, 50.0, -40.0],
    )
    result = select_scale(frame)
    assert result["selected"]["scale"] == 0.0
    assert result["selected"]["MAE"] == pytest.approx(result["baseline"]["MAE"])


def test_weighted_metrics_r2_moves_with_squared_error():
    actual = np.array([0.0, 10.0, 20.0, 100.0])
    base = weighted_metrics(actual, [0.0, 10.0, 20.0, 60.0], np.ones(4))
    better = weighted_metrics(actual, [0.0, 10.0, 20.0, 80.0], np.ones(4))
    assert better["RMSE"] < base["RMSE"]
    assert better["R2"] > base["R2"]


def test_meta_feature_contract_contains_no_outcome_or_residual_target():
    lowered = [name.lower() for name in META_FEATURES]
    assert not any("actual" in name for name in lowered)
    assert not any("residual" in name for name in lowered)
