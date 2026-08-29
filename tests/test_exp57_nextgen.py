import pandas as pd
from backend.app.ml.experiments.planned_date_reliability_exp57 import engineer_reliability
def test_repeated_schedule_revisions_reduce_reliability():
 f=pd.DataFrame({"canonical_project_id":["a"]*3,"snapshot_date":["2020-01-01","2020-02-01","2020-03-01"],"planned_completion_date":["2022-01-01"]*3,"revised_completion_date":["2022-01-01","2022-06-01","2023-01-01"]});o=engineer_reliability(f);assert o.iloc[-1]["exp57_schedule_revision_count"]>=2;assert o.iloc[-1]["exp57_planned_date_reliability"]<o.iloc[0]["exp57_planned_date_reliability"]
