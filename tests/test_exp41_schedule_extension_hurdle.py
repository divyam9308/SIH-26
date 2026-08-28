import pandas as pd
from backend.app.ml.experiments import adapter_exp41
from backend.app.ml.experiments.schedule_extension_hurdle_exp41 import add_extension_targets


def test_exp41_adapter_contract():
    assert adapter_exp41.EXPERIMENT_ID == "exp_41"
    assert adapter_exp41.EXPERIMENT_SEQUENCE == 41
    assert adapter_exp41.EXPERIMENT_SCOPE == "delay"


def test_extension_event_requires_material_positive_future_extension():
    frame = pd.DataFrame({"schedule_slippage_days": [100.0, 100.0, 100.0], "actual_delay_days": [120.0, 131.0, 80.0]})
    out = add_extension_targets(frame)
    assert out["exp41_extension_event"].tolist() == [0, 1, 0]
