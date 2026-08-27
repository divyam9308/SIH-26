from __future__ import annotations

import pandas as pd

from backend.app.ml.experiments.scope_semantics_exp21 import add_structured_scope_features, _decision


def test_scope_features_extract_engineering_signal() -> None:
    frame = pd.DataFrame({"project_name": ["Phase 2 construction of 14.5 km tunnel and bridge, 500 MW"]})
    out = add_structured_scope_features(frame).iloc[0]
    assert out.exp21_phase_number == 2
    assert out.exp21_length_km == 14.5
    assert out.exp21_capacity_mw == 500
    assert out.exp21_tunnel == 1
    assert out.exp21_bridge == 1
    assert out.exp21_numeric_tokens >= 3


def test_promotion_verdict_requires_no_target_regression() -> None:
    assert _decision(1.0, 0.0) == "PROMOTION CANDIDATE"
    assert _decision(2.0, -0.01) == "REGRESSION / DO NOT PROMOTE"
