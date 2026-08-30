import pytest

from backend.app.ml.experiments.audit_production_late_windows import WINDOWS, window_contract


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
