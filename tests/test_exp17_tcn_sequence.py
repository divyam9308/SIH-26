from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from backend.app.ml.experiments.adapters import default_experiment_adapter, get_experiment_adapter
from backend.app.ml.experiments.neural_sequence_common import HISTORY_VARIANTS, MIN_SEQUENCE, SEQUENCE_FEATURES, SequenceStore, rolling_folds
from backend.app.ml.experiments.tcn_sequence_exp17 import TCNForecaster


def _history(months: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2016-01-31", periods=months, freq="ME")
    return pd.DataFrame([
        {"canonical_project_id": "P1", "snapshot_date": stamp, "approved_cost_cr": 100.0,
         "revised_cost_cr": 100.0 + max(0, i - 20) * 0.8, "cumulative_expenditure_cr": 2.0 + i * 1.1,
         "physical_progress": min(100.0, i * 1.4), "schedule_slippage_days": max(0.0, (i - 24) * 4.0),
         "planned_duration_days": 1095.0, "expected_progress_percentage": min(100.0, i * 1.5)}
        for i, stamp in enumerate(dates)
    ])


def test_sequence_store_is_future_append_invariant():
    history = _history(24); cutoff = history.iloc[11].snapshot_date
    before = SequenceStore(history).raw("P1", cutoff, None)
    future = history.iloc[-1].copy(); future["snapshot_date"] = pd.Timestamp("2030-01-31"); future["revised_cost_cr"] = 999999.0
    after = SequenceStore(pd.concat([history, pd.DataFrame([future])], ignore_index=True)).raw("P1", cutoff, None)
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_all_history_variants_are_available():
    store = SequenceStore(_history(72)); stamp = pd.Timestamp("2021-12-31")
    assert [len(store.raw("P1", stamp, HISTORY_VARIANTS[k])) for k in ("12m", "24m", "36m", "60m", "full")] == [12, 24, 36, 60, 72]
    assert MIN_SEQUENCE == 3


def test_rolling_folds_are_forward_only():
    frame = pd.DataFrame([{"canonical_project_id": f"{year}-{p}", "completion_year": year} for year in range(2013, 2020) for p in range(12)])
    folds = rolling_folds(frame, max_folds=3); assert len(folds) == 3
    for fitting, validation in folds: assert max(fitting) < min(validation)


def test_tcn_forward_supports_variable_lengths():
    width = len(SEQUENCE_FEATURES) * 2
    model = TCNForecaster(width, [4, 5, 3])
    out = model(torch.zeros((2, 20, width)), torch.tensor([20, 7]), torch.zeros((2, 8)), torch.zeros((2, 3), dtype=torch.long))
    assert tuple(out.shape) == (2, 2)


def test_exp17_is_registered_as_default_challenger():
    adapter = get_experiment_adapter("exp_17")
    assert adapter.sequence == 17 and adapter.scope == "cost_delay"
    assert default_experiment_adapter().experiment_id == "exp_17"
