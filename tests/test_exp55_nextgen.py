import pandas as pd
from backend.app.ml.experiments.financial_burn_rate_exp55 import engineer_burn_rate
def test_future_report_does_not_change_earlier_burn_features():
 base=pd.DataFrame({"canonical_project_id":["a","a"],"snapshot_date":["2020-01-01","2020-02-01"],"cumulative_expenditure_cr":[10,20],"revised_cost_cr":[100,100],"approved_cost_cr":[100,100],"duration_ratio":[.2,.3]});early=engineer_burn_rate(base).iloc[0];ext=pd.concat([base,pd.DataFrame({"canonical_project_id":["a"],"snapshot_date":["2021-01-01"],"cumulative_expenditure_cr":[90],"revised_cost_cr":[110],"approved_cost_cr":[100],"duration_ratio":[1.2]})],ignore_index=True);later=engineer_burn_rate(ext).iloc[0];assert early["exp55_implied_months_to_revised_spend"]==later["exp55_implied_months_to_revised_spend"]
