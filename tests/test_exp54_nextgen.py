import pandas as pd
from backend.app.ml.experiments.portfolio_pressure_delay_exp54 import engineer_portfolio_pressure
def test_portfolio_pressure_counts_concurrent_projects():
 f=pd.DataFrame({"canonical_project_id":["a","b","c"],"snapshot_date":["2020-01-01"]*3,"implementing_agency":["A","A","B"],"sector":["S","S","S"],"schedule_slippage_days":[10,20,30],"cost_escalation_percentage":[1,2,3],"expenditure_ratio":[.2,.3,.4]});o=engineer_portfolio_pressure(f);assert list(o["exp54_agency_live_projects"])==[2,2,1];assert o["exp54_sector_live_projects"].eq(3).all()
