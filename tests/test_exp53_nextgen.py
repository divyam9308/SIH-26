import pandas as pd
from backend.app.ml.experiments.peer_pressure_cost_exp53 import engineer_peer_pressure,FEATURES
def test_peer_features_are_same_date_and_target_free():
 f=pd.DataFrame({"canonical_project_id":["a","b","c"],"snapshot_date":["2020-01-01"]*3,"sector":["x","x","y"],"cost_escalation_percentage":[1,2,3],"expenditure_ratio":[.2,.5,.9],"schedule_slippage_days":[10,20,30],"duration_ratio":[.5,.7,.9]});o=engineer_peer_pressure(f);assert o["exp53_peer_count"].eq(3).all();assert set(FEATURES).issubset(o.columns);assert o["exp53_cost_rank"].between(0,1).all()
