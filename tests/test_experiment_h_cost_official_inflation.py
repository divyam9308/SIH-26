from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.experiments.experiment_h_cost_official_inflation import (
    INFLATION_FEATURES,
    attach_inflation,
    load_official_index,
)


def test_missing_index_returns_none(tmp_path:Path):
    assert load_official_index(tmp_path/"missing.csv") is None


def test_index_requires_source_attribution(tmp_path:Path):
    p=tmp_path/"wpi.csv"
    pd.DataFrame({"month":["2020-01-01"],"index_value":[100],"source_url":[""]}).to_csv(p,index=False)
    with pytest.raises(ValueError): load_official_index(p)


def test_inflation_features_use_snapshot_and_approval_months(tmp_path:Path):
    p=tmp_path/"wpi.csv"
    dates=pd.date_range("2018-01-01",periods=30,freq="MS")
    pd.DataFrame({"month":dates,"index_value":[100+i for i in range(len(dates))],"source_url":["https://eaindustry.nic.in/source"]*len(dates)}).to_csv(p,index=False)
    idx=load_official_index(p)
    score=pd.DataFrame({"snapshot_date":["2020-01-15"],"approval_date":["2018-01-01"],"approved_cost_cr":[1000],"cost_escalation_percentage":[25]})
    out=attach_inflation(score,idx)
    for f in INFLATION_FEATURES: assert f in out
    assert out.loc[0,"exp_h_cumulative_since_approval_pct"] > 0
