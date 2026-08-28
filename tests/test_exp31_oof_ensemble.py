import numpy as np
from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.oof_ensemble_exp31 import EXPERIMENT_ID,FAMILIES,_weight_grid

def test_adapter_contract(): assert get_experiment_adapter(EXPERIMENT_ID).sequence==31
def test_convex_weight_grid_is_complete_and_valid():
    grid=_weight_grid(0.1); assert len(grid)==66; assert all(set(r)==set(FAMILIES) for r in grid); assert all(np.isclose(sum(r.values()),1.0) for r in grid); assert all(all(v>=0 for v in r.values()) for r in grid)
