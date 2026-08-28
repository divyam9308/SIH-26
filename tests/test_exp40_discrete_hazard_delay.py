import pandas as pd
from backend.app.ml.experiments import adapter_exp40
from backend.app.ml.experiments.discrete_hazard_delay_exp40 import expand_hazard_rows


def test_exp40_adapter_contract():
    assert adapter_exp40.EXPERIMENT_ID == "exp_40"
    assert adapter_exp40.EXPERIMENT_SEQUENCE == 40
    assert adapter_exp40.EXPERIMENT_SCOPE == "delay"


def test_hazard_expansion_marks_single_event():
    train = pd.DataFrame({
        "canonical_project_id": ["p1"],
        "snapshot_date": [pd.Timestamp("2020-01-01")],
        "completion_date": [pd.Timestamp("2020-07-01")],
        "sample_weight": [1.0],
        "x": [2.0],
    })
    out = expand_hazard_rows(train, ["x"])
    assert out["exp40_completion_event"].sum() == 1
    assert abs(out["sample_weight"].sum() - 1.0) < 1e-12
