import json

import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.residual_calibration_exp33 import (
    EXPERIMENT_ID,
    _corrections,
    _public_calibration,
    _weighted_median,
)


def test_adapter_contract():
    adapter = get_experiment_adapter(EXPERIMENT_ID)
    assert adapter.sequence == 33
    assert callable(adapter.module.fit_against_production)
    assert callable(adapter.module.filter_comparable_rows)
    assert callable(adapter.module.predict_project)


def test_weighted_median_and_calibration_fallbacks():
    assert _weighted_median([0, 10, 20], [1, 10, 1]) == 10.0
    calibration = {
        "edges": [-np.inf, 5.0, np.inf],
        "global_median": 1.0,
        "bin_medians": {0: 2.0, 1: 3.0},
        "stage_bin_medians": {("early", 0): 4.0},
    }
    frame = pd.DataFrame({"lifecycle_stage": ["early", "late"]})
    corrections = _corrections(frame, np.array([2.0, 8.0]), calibration)
    assert corrections.tolist() == [4.0, 3.0]


def test_public_calibration_is_strict_json_safe_without_changing_runtime_edges():
    runtime = {
        "edges": [-np.inf, -2.5, 7.25, np.inf],
        "global_median": 1.5,
        "bin_medians": {0: 1.0},
        "stage_bin_medians": {},
        "oof_rows": 42,
        "fold_years": [2016, 2017, 2018],
    }
    public = _public_calibration(runtime)

    # Keep unbounded numeric sentinels in runtime for np.digitize.
    assert np.isneginf(runtime["edges"][0])
    assert np.isposinf(runtime["edges"][-1])

    # Public diagnostics use JSON null for open-ended edges and must serialize
    # under the same strict contract used by the Actions result writer.
    assert public["edges"] == [None, -2.5, 7.25, None]
    assert public["edge_semantics"] == "null first/last edge means unbounded"
    json.dumps(public, allow_nan=False)
