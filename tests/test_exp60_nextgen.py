import pandas as pd
from backend.app.ml.experiments.quality_fallback_exp60 import engineer_quality
def test_quality_flags_future_approval_and_missing_plan():
 f=pd.DataFrame({"snapshot_date":["2020-01-01"],"approval_date":["2096-01-01"],"planned_completion_date":[None],"approved_cost_cr":[100],"revised_cost_cr":[100],"cumulative_expenditure_cr":[10],"schedule_slippage_days":[0],"duration_ratio":[.5],"cost_escalation_percentage":[0],"expenditure_ratio":[.1]});o=engineer_quality(f);assert o.loc[0,"exp60_approval_date_future"]==1;assert o.loc[0,"exp60_planned_date_missing"]==1;assert 0<o.loc[0,"exp60_quality_score"]<1
