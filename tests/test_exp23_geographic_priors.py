from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments.geographic_priors_exp23 import project_level_residuals, _prior, corrections, _decision


def test_one_residual_per_project_and_missing_geo_zero_correction() -> None:
    train = pd.DataFrame({
        "canonical_project_id": ["A", "A", "B"],
        "state": ["X", "X", "Y"],
        "sector": ["Road", "Road", "Power"],
        "actual_delay_days": [10.0, 20.0, 30.0],
    })
    projects = project_level_residuals(train, np.array([5.0, 5.0, 10.0]), "actual_delay_days")
    assert len(projects) == 2
    prior = _prior(projects, "state", 1.0, 240.0)
    test = pd.DataFrame({"state": ["X", None], "sector": ["Road", "Road"]})
    corr = corrections(test, prior, {}, 240.0)
    assert corr[0] != 0
    assert corr[1] == 0


def test_promotion_verdict_is_conservative() -> None:
    assert _decision(1.0, 1.0) == "PROMOTION CANDIDATE"
    assert _decision(1.0, -0.001) == "REGRESSION / DO NOT PROMOTE"
