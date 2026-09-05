import numpy as np
import pandas as pd

from backend.app.ml.experiments import exp_delay_route_stage_recalibration as exp


def test_weighted_median_and_grids():
    assert exp._weighted_median(np.array([1.0, 5.0, 9.0]), np.array([1.0, 3.0, 1.0])) == 5.0
    assert exp.BETAS == (0.0, 0.5, 1.0, 1.5)
    assert exp.LAMBDAS == (20.0, 40.0, 80.0)


def test_stage_is_outcome_free():
    frame = pd.DataFrame({"duration_ratio": [0.2, 0.7, 1.0, 1.4, np.nan]})
    assert exp._stage(frame).astype(str).tolist() == ["early", "mid", "late", "very_late", "missing"]
