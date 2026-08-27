from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.obstruction_reasons_exp24 import categorize_obstruction_reason, reason_text


def test_reason_categories_are_deterministic() -> None:
    flags = categorize_obstruction_reason("Delay due to land acquisition, forest clearance and contractor mobilisation")
    assert flags["land_acquisition"] == 1
    assert flags["clearance"] == 1
    assert flags["contractor"] == 1
    assert flags["funding"] == 0


def test_numeric_status_is_not_misrepresented_as_reason_text() -> None:
    frame = pd.DataFrame({"current_schedule_status": ["delayed"], "project_name": ["Example project"]})
    text, columns = reason_text(frame)
    assert columns == []
    assert text.iloc[0] == ""
