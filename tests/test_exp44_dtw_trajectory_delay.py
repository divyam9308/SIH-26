import numpy as np
from backend.app.ml.experiments import adapter_exp44
from backend.app.ml.experiments.dtw_trajectory_delay_exp44 import dtw_distance


def test_exp44_adapter_contract():
    assert adapter_exp44.EXPERIMENT_ID == "exp_44"
    assert adapter_exp44.EXPERIMENT_SEQUENCE == 44
    assert adapter_exp44.EXPERIMENT_SCOPE == "delay"


def test_dtw_identical_sequence_has_zero_distance():
    seq = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert dtw_distance(seq, seq) == 0.0
