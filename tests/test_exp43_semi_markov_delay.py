import pandas as pd
from backend.app.ml.experiments import adapter_exp43
from backend.app.ml.experiments.semi_markov_delay_exp43 import observable_state


def test_exp43_adapter_contract():
    assert adapter_exp43.EXPERIMENT_ID == "exp_43"
    assert adapter_exp43.EXPERIMENT_SEQUENCE == 43
    assert adapter_exp43.EXPERIMENT_SCOPE == "delay"


def test_observable_states_use_only_as_of_signals():
    frame = pd.DataFrame({
        "lifecycle_stage": ["mid", "late", "early"],
        "schedule_slippage_days": [0.0, 120.0, 500.0],
        "progress_deviation": [0.0, -20.0, -40.0],
    })
    assert observable_state(frame).tolist() == ["mid|healthy", "late|stressed", "early|severe"]
