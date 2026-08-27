from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from backend.app.ml.experiments.adapters import default_experiment_adapter, get_experiment_adapter
from backend.app.ml.experiments.gru_sequence_exp16 import (
    GRUForecaster,
    HISTORY_VARIANTS,
    MIN_SEQUENCE,
    SEQUENCE_FEATURES,
    SequenceStore,
    _rolling_folds,
)


def _history(months: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2016-01-31", periods=months, freq="ME")
    rows = []
    for i, stamp in enumerate(dates):
        rows.append({
            "canonical_project_id": "P1",
            "snapshot_date": stamp,
            "approved_cost_cr": 100.0,
            "revised_cost_cr": 100.0 + max(0, i - 20) * 0.8,
            "cumulative_expenditure_cr": 2.0 + i * 1.1,
            "physical_progress": min(100.0, i * 1.4),
            "schedule_slippage_days": max(0.0, (i - 24) * 4.0),
            "planned_duration_days": 1095.0,
            "expected_progress_percentage": min(100.0, i * 1.5),
        })
    return pd.DataFrame(rows)


def test_sequence_store_uses_only_reports_available_as_of_prediction_date():
    history = _history(24)
    cutoff = history.iloc[11].snapshot_date
    before = SequenceStore(history).raw("P1", cutoff, None)
    future = history.iloc[-1].copy()
    future["snapshot_date"] = pd.Timestamp("2030-01-31")
    future["revised_cost_cr"] = 999999.0
    after = SequenceStore(pd.concat([history, pd.DataFrame([future])], ignore_index=True)).raw("P1", cutoff, None)
    assert before is not None and len(before) == 12
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_history_variants_truncate_only_the_past_not_the_future():
    store = SequenceStore(_history(72))
    stamp = pd.Timestamp("2021-12-31")
    assert len(store.raw("P1", stamp, HISTORY_VARIANTS["12m"])) == 12
    assert len(store.raw("P1", stamp, HISTORY_VARIANTS["24m"])) == 24
    assert len(store.raw("P1", stamp, HISTORY_VARIANTS["36m"])) == 36
    assert len(store.raw("P1", stamp, HISTORY_VARIANTS["60m"])) == 60
    assert len(store.raw("P1", stamp, HISTORY_VARIANTS["full"])) == 72
    assert MIN_SEQUENCE == 3


def test_rolling_folds_are_forward_only():
    rows = []
    for year in range(2013, 2020):
        for p in range(12):
            rows.append({"canonical_project_id": f"{year}-{p}", "completion_year": year})
    folds = _rolling_folds(pd.DataFrame(rows), max_folds=3)
    assert len(folds) == 3
    for fitting, validation in folds:
        assert max(fitting) < min(validation)
        assert len(validation) == 1


def test_gru_forward_supports_variable_length_sequences():
    model = GRUForecaster(sequence_width=len(SEQUENCE_FEATURES) * 2, cardinalities=[4, 5, 3])
    sequence = torch.zeros((2, 10, len(SEQUENCE_FEATURES) * 2), dtype=torch.float32)
    lengths = torch.tensor([10, 6], dtype=torch.long)
    numeric = torch.zeros((2, 8), dtype=torch.float32)
    cats = torch.zeros((2, 3), dtype=torch.long)
    out = model(sequence, lengths, numeric, cats)
    assert tuple(out.shape) == (2, 2)


def test_exp16_adapter_is_registered_as_default_challenger():
    adapter = get_experiment_adapter("exp_16")
    assert adapter.sequence == 16
    assert adapter.scope == "cost_delay"
    assert default_experiment_adapter().experiment_id == "exp_16"
