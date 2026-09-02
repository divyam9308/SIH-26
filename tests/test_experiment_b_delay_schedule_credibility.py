import numpy as np
import pandas as pd

from backend.app.ml.experiments.experiment_b_delay_schedule_credibility import (
    PRIOR_FEATURES,
    attach_schedule_credibility,
)


def _reference():
    rows = []
    for i in range(30):
        completion = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 10)
        revised = completion - pd.Timedelta(days=100 + (i % 5) * 20)
        rows.append(
            {
                "canonical_project_id": f"P{i}",
                "implementing_agency": "Agency A" if i < 20 else "Agency B",
                "sector": "Roads" if i % 2 else "Railways",
                "snapshot_date": revised - pd.Timedelta(days=300),
                "revised_completion_date": revised,
                "completion_date": completion,
                "planned_duration_days": 1000,
                "actual_delay_days": i * 1000,
            }
        )
    return pd.DataFrame(rows)


def test_holdout_target_cannot_change_credibility_features():
    reference = _reference()
    score = reference.iloc[:5].copy()
    score["actual_delay_days"] = [1, 2, 3, 4, 5]
    first = attach_schedule_credibility(reference, score)
    score["actual_delay_days"] = [999999] * len(score)
    second = attach_schedule_credibility(reference, score)
    for feature in PRIOR_FEATURES:
        assert feature in first
        assert np.allclose(
            pd.to_numeric(first[feature], errors="coerce").fillna(-9999),
            pd.to_numeric(second[feature], errors="coerce").fillna(-9999),
        )


def test_reference_target_is_completion_date_not_delay_label():
    reference = _reference()
    score = reference.iloc[:3].copy()
    first = attach_schedule_credibility(reference, score)
    changed = reference.copy()
    changed["actual_delay_days"] = np.arange(len(changed)) * 999999
    second = attach_schedule_credibility(changed, score)
    assert np.allclose(first["exp_b_schedule_bias_days"], second["exp_b_schedule_bias_days"])
