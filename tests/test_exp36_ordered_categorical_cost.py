import pandas as pd
from backend.app.ml.experiments import adapter_exp36
from backend.app.ml.experiments.ordered_categorical_cost_exp36 import enrich_context


def test_exp36_adapter_contract():
    assert adapter_exp36.EXPERIMENT_ID == "exp_36"
    assert adapter_exp36.EXPERIMENT_SEQUENCE == 36
    assert adapter_exp36.EXPERIMENT_SCOPE == "cost"


def test_context_interactions_do_not_use_raw_project_name():
    frame = pd.DataFrame({"sector": ["Rail"], "ministry": ["M"], "implementing_agency": ["A"], "state": ["S"], "lifecycle_stage": ["mid"], "project_name": ["Unique Project"]})
    out = enrich_context(frame)
    assert out.loc[0, "exp36_agency_sector"] == "A||Rail"
    assert out.loc[0, "exp36_state_sector"] == "S||Rail"
    assert not any(name.startswith("project_name") for name in ["exp36_agency_sector", "exp36_state_sector", "exp36_ministry_sector"])
