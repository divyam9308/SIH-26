import pandas as pd
from backend.app.ml.experiments.nextgen_common import shrunk_calibration
def test_shrunk_cost_calibration_is_finite_and_shrinks():
    rows=pd.DataFrame({"prediction":[0,1,2,3,4,5]*10,"residual":[10]*30+[-10]*30,"sample_weight":[.1]*60,"lifecycle_stage":["early"]*30+["late"]*30})
    cal=shrunk_calibration(rows,strength=40)
    assert cal["oof_rows"]==60
    assert all(pd.notna(v) for v in cal["bin_medians"].values())
    assert cal["strength"]==40
