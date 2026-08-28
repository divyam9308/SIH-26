from __future__ import annotations
import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.remaining_delay_exp27 import EXPERIMENT_ID, _eligible


def test_adapter_contract():
    adapter = get_experiment_adapter(EXPERIMENT_ID)
    assert adapter.sequence == 27
    assert adapter.scope == "delay"


def test_remaining_delay_target_is_signed_and_reweighted():
    frame = pd.DataFrame({
        "canonical_project_id": ["A", "A", "B"],
        "actual_delay_days": [100.0, 80.0, 50.0],
        "schedule_slippage_days": [70.0, 90.0, 10.0],
        "sample_weight": [99.0, 99.0, 99.0],
    })
    out = _eligible(frame)
    assert out.exp27_future_schedule_error.tolist() == [30.0, -10.0, 40.0]
    totals = out.groupby("canonical_project_id").sample_weight.sum()
    assert np.allclose(totals.to_numpy(dtype=float), 1.0)
