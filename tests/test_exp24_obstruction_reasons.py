from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments.obstruction_reasons_exp24 import (
    CATEGORY_PATTERNS,
    _corrections,
    add_asof_obstruction_features,
    categorize_obstruction_reason,
    match_history_to_projects,
    reason_text,
)


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


def test_official_reason_only_appears_after_publication_date() -> None:
    frame = pd.DataFrame({
        "canonical_project_id": ["P1", "P1"],
        "project_name": ["Bongaigaon TPP", "Bongaigaon TPP"],
        "state": ["Assam", "Assam"],
        "sector": ["Power", "Power"],
        "snapshot_date": pd.to_datetime(["2017-12-31", "2018-06-30"]),
    })
    reason = "heavy monsoon and poor contractor performance"
    row = {
        "source_publication_date": pd.Timestamp("2018-04-05"),
        "source_type": "Lok Sabha",
        "source_url": "https://example.gov.in/official.pdf",
        "project_name_hint": "Bongaigaon TPP",
        "state": "Assam",
        "sector_hint": "Power",
        "reason_text": reason,
        "source_row_id": 0,
    }
    row.update(categorize_obstruction_reason(reason))
    history = pd.DataFrame([row])
    matched, audit = match_history_to_projects(frame, history)
    assert audit["matched_unique_projects"] == 1
    enriched = add_asof_obstruction_features(frame, matched)
    assert enriched.iloc[0].exp24_reason_observed == 0
    assert enriched.iloc[1].exp24_reason_observed == 1
    assert enriched.iloc[1].exp24_contractor == 1
    assert enriched.iloc[1].exp24_geology_weather == 1


def test_unmatched_or_unobserved_snapshot_gets_zero_correction() -> None:
    frame = pd.DataFrame({
        "exp24_reason_observed": [0.0, 1.0],
        **{f"exp24_{name}": [0.0, 0.0] for name in CATEGORY_PATTERNS},
    })
    frame.loc[1, "exp24_land_acquisition"] = 1.0
    correction = _corrections(frame, {"exp24_land_acquisition": 60.0}, 180.0)
    assert np.isclose(correction[0], 0.0)
    assert np.isclose(correction[1], 60.0)
