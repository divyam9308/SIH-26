import numpy as np
import pandas as pd

from backend.app.ml.experiments.experiment_c_delay_dynamic_latent_state import (
    STATE_FEATURES,
    attach_dynamic_state,
)


def _history():
    dates = pd.date_range("2018-01-01", periods=8, freq="90D")
    return pd.DataFrame(
        {
            "canonical_project_id": ["P1"] * len(dates),
            "snapshot_date": dates,
            "physical_progress": np.linspace(5, 70, len(dates)),
            "expenditure_ratio": np.linspace(0.04, 0.75, len(dates)),
            "schedule_slippage_days": np.linspace(0, 240, len(dates)),
            "duration_ratio": np.linspace(0.2, 1.1, len(dates)),
            "cost_escalation_percentage": np.linspace(0, 20, len(dates)),
            "progress_deviation": np.linspace(-5, -25, len(dates)),
        }
    )


def test_appending_future_report_cannot_change_earlier_state():
    reference = pd.concat([_history(), _history().assign(canonical_project_id="P2")], ignore_index=True)
    score = _history()
    earlier = attach_dynamic_state(reference, score)
    future = score.iloc[[-1]].copy()
    future["snapshot_date"] = pd.Timestamp("2030-01-01")
    future["schedule_slippage_days"] = 99999
    future["physical_progress"] = 0
    extended = pd.concat([score, future], ignore_index=True)
    after = attach_dynamic_state(reference, extended).iloc[: len(score)]
    for feature in STATE_FEATURES:
        assert np.allclose(earlier[feature], after[feature], equal_nan=True)


def test_state_features_are_finite_on_complete_history():
    reference = pd.concat([_history(), _history().assign(canonical_project_id="P2")], ignore_index=True)
    enriched = attach_dynamic_state(reference, _history())
    for feature in STATE_FEATURES:
        assert feature in enriched
        assert np.isfinite(pd.to_numeric(enriched[feature], errors="coerce")).all()
