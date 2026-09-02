import numpy as np
import pandas as pd

from backend.app.ml.experiments.experiment_a_delay_execution_clock import (
    CLOCK_FEATURES,
    attach_clock,
)


def _frame():
    rows = []
    for i in range(60):
        ratio = 0.2 + (i % 20) * 0.05
        rows.append(
            {
                "canonical_project_id": f"P{i//3}",
                "duration_ratio": ratio,
                "physical_progress": min(100.0, ratio * 70.0),
                "expenditure_ratio": min(1.4, ratio * 0.75),
                "sector": "Railways" if i % 2 else "Roads",
                "_norm_sector": "railways" if i % 2 else "roads",
                "actual_delay_days": 99999 + i,
            }
        )
    return pd.DataFrame(rows)


def test_clock_features_do_not_use_delay_target():
    reference = _frame()
    score = reference.iloc[:8].copy()
    first = attach_clock(reference, score)
    changed = reference.copy()
    changed["actual_delay_days"] = np.arange(len(changed)) * 1000000
    second = attach_clock(changed, score)
    for feature in CLOCK_FEATURES:
        assert feature in first.columns
        assert np.allclose(
            pd.to_numeric(first[feature], errors="coerce").fillna(-9999),
            pd.to_numeric(second[feature], errors="coerce").fillna(-9999),
        )


def test_clock_preserves_score_row_count():
    reference = _frame()
    score = reference.iloc[:11].copy()
    enriched = attach_clock(reference, score)
    assert len(enriched) == len(score)
