import pandas as pd
import pytest

from backend.app.ml.experiments.audit_production_late_windows import (
    WINDOWS,
    select_aft_projects_for_one_off_audit,
    window_contract,
)
from backend.app.ml.production_exp35_baseline import _select_aft_calibration_projects


def test_fixed_one_off_windows_are_exact():
    assert WINDOWS == {
        2021: (2022, 2025),
        2022: (2023, 2025),
        2023: (2024, 2025),
    }
    assert window_contract(2021) == (2022, 2025)
    assert window_contract(2022) == (2023, 2025)
    assert window_contract(2023) == (2024, 2025)


def test_other_cutoffs_are_not_part_of_this_audit():
    with pytest.raises(ValueError):
        window_contract(2019)
    with pytest.raises(ValueError):
        window_contract(2024)


def test_one_off_selector_uses_all_available_aft_projects_when_below_688():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["A", "A", "B", "C"],
            "snapshot_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01"]
            ),
            "planned_completion_date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-06-01", None]
            ),
        }
    )

    # The real production selector stays strict.
    with pytest.raises(RuntimeError, match="cannot form the requested 688-project"):
        _select_aft_calibration_projects(frame)

    # PR #109 alone relaxes the historical fixed-size gate to available evidence.
    assert select_aft_projects_for_one_off_audit(frame) == {"A", "B"}


def test_one_off_selector_preserves_normal_selection_when_limit_is_available():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["A", "A", "B"],
            "snapshot_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
            "planned_completion_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-06-01"]),
        }
    )
    assert select_aft_projects_for_one_off_audit(frame, limit=1) == {"A"}
