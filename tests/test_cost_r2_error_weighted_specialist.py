import numpy as np
import pandas as pd

from backend.app.ml.experiments.cost_r2_error_weighted_specialist import (
    MAX_WEIGHT_MULTIPLIER,
    error_weight_multiplier,
    project_error_scores,
    select_strength_and_alpha,
)
from backend.app.ml.experiments.cost_r2_oof_sharding import OOF_YEARS, load_oof_shards


def test_project_error_scores_use_squared_production_oof_error():
    oof = pd.DataFrame({
        "canonical_project_id": ["a", "a", "b"],
        "production_residual": [2.0, -4.0, 1.0],
    })
    scores = project_error_scores(oof)
    assert scores["a"] == 10.0
    assert scores["b"] == 1.0


def test_error_multiplier_upweights_high_error_projects_and_caps():
    prior = pd.DataFrame({
        "canonical_project_id": ["a", "b", "c"],
        "production_residual": [1.0, 2.0, 20.0],
    })
    frame = pd.DataFrame({"canonical_project_id": ["a", "b", "c", "unseen"]})
    mult = error_weight_multiplier(frame, prior, strength=2.0)
    assert mult[2] >= mult[1] >= mult[0]
    assert mult[3] == 1.0
    assert mult.max() <= MAX_WEIGHT_MULTIPLIER


def test_weighted_selector_obeys_mae_constraint():
    meta = pd.DataFrame({
        "actual_cost_overrun_percentage": [0.0, 0.0, 10.0, 100.0],
        "production_prediction": [0.0, 0.0, 10.0, 60.0],
        "unweighted_specialist_prediction": [5.0, 5.0, 15.0, 70.0],
        "sample_weight": np.ones(4),
    })
    weighted = {
        0.5: np.array([0.0, 0.0, 10.0, 90.0]),
        1.0: np.array([20.0, 20.0, 30.0, 120.0]),
    }
    result = select_strength_and_alpha(meta, weighted)
    selected = result["selected"]
    assert selected["strength"] == 0.5
    assert selected["alpha"] > 0
    assert selected["MAE"] <= result["baseline"]["MAE"]
    assert selected["RMSE"] < result["baseline"]["RMSE"]


def test_oof_shards_recombine_only_the_frozen_four_year_contract(tmp_path):
    for year in OOF_YEARS:
        frame = pd.DataFrame({
            "canonical_project_id": [f"p-{year}"],
            "actual_cost_overrun_percentage": [10.0],
            "sample_weight": [1.0],
            "production_prediction": [9.0],
            "production_residual": [1.0],
            "oof_year": [year],
        })
        frame.to_pickle(tmp_path / f"production_oof_{year}.pkl")

    combined = load_oof_shards(tmp_path)
    years = tuple(sorted(pd.to_numeric(combined["oof_year"]).astype(int).unique()))
    assert years == OOF_YEARS
    assert len(combined) == len(OOF_YEARS)
