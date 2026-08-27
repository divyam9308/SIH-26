from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backend.app.ml.experiments import adapter_exp5
from backend.app.ml.experiments.adapters import discover_experiment_adapters


def test_exp5_is_registered_as_challenger():
    discovered = {item.experiment_id: item for item in discover_experiment_adapters()}
    assert "exp_5" in discovered
    assert discovered["exp_5"].sequence == 5
    assert discovered["exp_5"].scope == "cost_delay"


def test_exp5_comparable_rows_are_fixed_to_2022_2025_and_common_projects():
    frame = pd.DataFrame([
        {"canonical_project_id": "A", "completion_year": 2021},
        {"canonical_project_id": "B", "completion_year": 2022},
        {"canonical_project_id": "C", "completion_year": 2025},
        {"canonical_project_id": "D", "completion_year": 2026},
        {"canonical_project_id": "X", "completion_year": 2024},
    ])
    state = {"test_period": (2022, 2025), "common_projects": {"B", "C"}}
    result = adapter_exp5.filter_comparable_rows(frame, state)
    assert result.canonical_project_id.tolist() == ["B", "C"]


def test_exp5_project_prediction_returns_cost_and_delay():
    class Model:
        def __init__(self, value):
            self.value = value

        def predict(self, frame):
            return [self.value]

    state = {
        "models": {"cost": Model(12.5), "delay": Model(123.0), "risk": Model("HIGH")},
        "features": ["f"],
        "selected_algorithms": {"cost": "extra_trees", "delay": "extra_trees"},
        "seed": 26519,
        "test_period": (2022, 2025),
    }
    payload = adapter_exp5.predict_project(pd.Series({"f": 1.0}), state)
    assert payload["predicted_cost_overrun"] == 12.5
    assert payload["predicted_delay_days"] == 123.0
    assert payload["predicted_risk"] == "HIGH"
    assert payload["comparison_test_period"] == [2022, 2025]
