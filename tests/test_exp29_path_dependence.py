from __future__ import annotations
import pandas as pd
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.path_dependence_exp29 import EXPERIMENT_ID, PATH_FEATURES, _path_history

def test_adapter_contract():
    assert get_experiment_adapter(EXPERIMENT_ID).sequence == 29

def test_path_features_are_as_of_and_revision_counts_accumulate():
    frame = pd.DataFrame({"canonical_project_id": ["A"] * 4, "snapshot_date": pd.to_datetime(["2020-01-01","2020-02-01","2020-03-01","2020-04-01"]), "revised_cost_cr": [100.0,100.0,120.0,110.0], "cost_escalation_percentage": [0.0,0.0,20.0,10.0], "schedule_slippage_days": [0.0,10.0,20.0,15.0]})
    a = _path_history(frame)
    assert a.exp29_cost_revision_count.tolist() == [0,0,1,2]
    assert a.exp29_schedule_revision_count.tolist() == [0,1,2,3]
    changed = frame.copy(); changed.loc[3, "revised_cost_cr"] = 9999.0
    b = _path_history(changed)
    for feature in PATH_FEATURES:
        assert a.loc[:2, feature].equals(b.loc[:2, feature])
