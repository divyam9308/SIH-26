import pandas as pd
from backend.app.ml.experiments.nextgen_common import normalize_taxonomy
def test_taxonomy_normalization_is_deterministic():
 f=pd.DataFrame({"sector":[" Roads & Bridges ","ROADS--BRIDGES"],"implementing_agency":["NHAI Ltd.","NHAI LTD"]});o=normalize_taxonomy(f);assert o["_norm_sector"].nunique()==1;assert o["_norm_implementing_agency"].nunique()==1
