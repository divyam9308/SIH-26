import numpy as np
import pandas as pd
import pytest

from backend.app.ml.production_delay_baseline import (
    DEFAULT_PRODUCTION_WINDOW,
    PATH_FEATURES,
    PRODUCTION_DELAY_EVALUATION_COHORT,
    ProductionDelayBlendModel,
    enrich_history_for_delay_production,
)
from backend.app.services import lifecycle_retraining_service, monthly_prediction_service


class _ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def test_exp34_production_window_uses_evidence_defined_comparable_cohort():
    assert DEFAULT_PRODUCTION_WINDOW == "2001_2021"
    assert monthly_prediction_service.DEFAULT_PRODUCTION_WINDOW == "2001_2021"
    assert PRODUCTION_DELAY_EVALUATION_COHORT == "shared_exp12_comparable_evidence_cohort"
    assert lifecycle_retraining_service.train_window_with_promoted_cost_and_delay.__name__ == (
        "train_window_with_promoted_cost_and_delay"
    )


def test_production_delay_blend_predicts_weighted_ensemble_only():
    models = {
        "extra_trees": _ConstantModel(100.0),
        "lightgbm": _ConstantModel(200.0),
        "xgboost": _ConstantModel(300.0),
    }
    model = ProductionDelayBlendModel(
        models=models,
        weights={"extra_trees": 0.2, "lightgbm": 0.6, "xgboost": 0.2},
        features=["feature"],
    )
    prediction = model.predict(pd.DataFrame({"feature": [1.0, 2.0]}))
    assert prediction.tolist() == pytest.approx([200.0, 200.0])


def test_production_inference_enrichment_contains_all_exp34_path_features():
    history = pd.DataFrame(
        {
            "canonical_project_id": ["A", "A", "A"],
            "snapshot_date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
            "revised_cost_cr": [100.0, 110.0, 110.0],
            "cost_escalation_percentage": [0.0, 10.0, 10.0],
            "schedule_slippage_days": [0.0, 30.0, 20.0],
        }
    )
    enriched = enrich_history_for_delay_production(history)
    assert set(PATH_FEATURES).issubset(enriched.columns)
    assert enriched.loc[enriched.index[-1], "exp34_cost_revision_count"] == 1
    assert enriched.loc[enriched.index[-1], "exp34_schedule_revision_count"] == 2
