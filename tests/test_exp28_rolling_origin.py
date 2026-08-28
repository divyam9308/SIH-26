from __future__ import annotations
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.rolling_origin_exp28 import EXPERIMENT_ID, _rolling_folds


def test_adapter_contract():
    adapter = get_experiment_adapter(EXPERIMENT_ID)
    assert adapter.sequence == 28
    assert adapter.scope == "cost+delay"


def test_rolling_folds_are_strictly_forward():
    rows = []
    for year in range(2015, 2022):
        for project in range(4):
            rows.append({"completion_year": year, "canonical_project_id": f"{year}-{project}"})
    frame = pd.DataFrame(rows)
    folds = _rolling_folds(frame, max_folds=3)
    assert len(folds) == 3
    for fitting, validation, year in folds:
        assert fitting.completion_year.max() < year
        assert set(validation.completion_year) == {year}
