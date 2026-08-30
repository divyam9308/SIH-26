import pandas as pd
from backend.app.ml.experiments.u4_cross_target_exp65 import _aligned

def test_cross_target_alignment_uses_exact_snapshot_keys():
    c=pd.DataFrame({"canonical_project_id":["a","b"],"snapshot_date":pd.to_datetime(["2020-01-01","2020-02-01"]),"production_prediction":[1.,2.],"residual":[.1,.2],"sample_weight":[1.,1.]})
    d=pd.DataFrame({"canonical_project_id":["b","a"],"snapshot_date":pd.to_datetime(["2020-02-01","2020-01-01"]),"production_prediction":[20.,10.],"residual":[2.,1.]})
    out=_aligned(c,d).sort_values("canonical_project_id")
    assert out["cost_prediction"].tolist()==[1.,2.]
    assert out["delay_prediction"].tolist()==[10.,20.]
    assert len(out)==2
