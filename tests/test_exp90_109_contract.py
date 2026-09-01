import importlib,importlib.util
import pytest
from backend.app.ml.experiments.post_u1_cost_common import window_contract

def _adapter():
    found=[]
    for seq in range(90,110):
        name=f'backend.app.ml.experiments.adapter_exp{seq}'
        if importlib.util.find_spec(name) is not None: found.append(importlib.import_module(name))
    assert len(found)==1
    return found[0]

def test_new_cost_experiment_contract():
    a=_adapter();assert 90<=a.EXPERIMENT_SEQUENCE<=109;assert a.EXPERIMENT_SCOPE=='cost';assert a.PROMOTION_ALLOWED is False;assert callable(a.fit_experiment)
    assert window_contract(2019)==(2020,2025,949,14847);assert window_contract(2021)==(2022,2025,721,11200)
    with pytest.raises(ValueError): window_contract(2022)
