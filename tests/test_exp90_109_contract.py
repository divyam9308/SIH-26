import importlib
import importlib.util

import pandas as pd
import pytest

from backend.app.ml.experiments.post_u1_cost_common import window_contract


def _adapter():
    found = []
    for seq in range(90, 110):
        name = f"backend.app.ml.experiments.adapter_exp{seq}"
        if importlib.util.find_spec(name) is not None:
            found.append(importlib.import_module(name))
    assert len(found) == 1
    return found[0]


def test_new_cost_experiment_contract():
    adapter = _adapter()
    assert 90 <= adapter.EXPERIMENT_SEQUENCE <= 109
    assert adapter.EXPERIMENT_SCOPE == "cost"
    assert adapter.PROMOTION_ALLOWED is False
    assert callable(adapter.fit_experiment)
    assert window_contract(2019) == (2020, 2025, 949, 14847)
    assert window_contract(2021) == (2022, 2025, 721, 11200)
    with pytest.raises(ValueError):
        window_contract(2022)


def test_exp92_state_handles_noncontiguous_cohort_index():
    adapter = _adapter()
    if adapter.EXPERIMENT_SEQUENCE != 92:
        pytest.skip("Exp92-specific regression test")

    from backend.app.ml.experiments.exp92_state_space_cost import _state

    frame = pd.DataFrame(
        {
            "canonical_project_id": ["A", "B", "A", "B"],
            "snapshot_date": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"],
            "cost_escalation_percentage": [10.0, 20.0, 14.0, 18.0],
        },
        index=[101, 205, 999, 1403],
    )

    engineered = _state(frame)

    assert len(engineered) == len(frame)
    assert engineered["canonical_project_id"].tolist() == frame["canonical_project_id"].tolist()
    assert engineered["exp92_level"].notna().all()
    assert engineered["exp92_terminal_projection"].notna().all()
