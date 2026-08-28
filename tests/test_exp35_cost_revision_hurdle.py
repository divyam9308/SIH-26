import pandas as pd

from backend.app.ml.experiments import adapter_exp35
from backend.app.ml.experiments.cost_revision_hurdle_exp35 import _add_hurdle_targets


def test_exp35_adapter_contract():
    assert adapter_exp35.EXPERIMENT_ID == "exp_35"
    assert adapter_exp35.EXPERIMENT_SEQUENCE == 35
    assert adapter_exp35.EXPERIMENT_SCOPE == "cost"
    assert callable(adapter_exp35.fit_against_production)
    assert callable(adapter_exp35.filter_comparable_rows)
    assert callable(adapter_exp35.predict_project)


def test_hurdle_target_uses_only_current_and_final_cost_values():
    frame = pd.DataFrame({
        "cost_escalation_percentage": [10.0, 20.0, 30.0],
        "actual_cost_overrun_percentage": [10.4, 24.0, 27.0],
    })
    out = _add_hurdle_targets(frame)
    assert out["exp35_revision_event"].tolist() == [0, 1, 1]
    assert out["exp35_remaining_revision"].round(4).tolist() == [0.4, 4.0, -3.0]
