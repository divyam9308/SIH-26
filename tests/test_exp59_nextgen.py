import numpy as np,pandas as pd
from backend.app.ml.experiments.pca_path_scores_exp59 import engineer_pca,FEATURES,CANDIDATES
def test_pca_scores_fit_training_rows_and_are_finite():
 rows=[]
 for i in range(12):
  row={"canonical_project_id":str(i),"snapshot_date":"2020-01-01","completion_year":2020 if i<10 else 2023}
  for j,c in enumerate(CANDIDATES[:5]):row[c]=i+j
  rows.append(row)
 o=engineer_pca(pd.DataFrame(rows));assert set(FEATURES).issubset(o.columns);assert np.isfinite(o[FEATURES].to_numpy()).all()
