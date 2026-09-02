from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.experiments.experiment_j_cost_tender_economics import (
    attach_tender_economics,
    load_verified_tenders,
)


def _write(path:Path):
    pd.DataFrame({
        "canonical_project_id":["P1","P1"],
        "award_date":["2020-01-01","2021-01-01"],
        "estimate_cr":[100,200],
        "award_value_cr":[90,260],
        "package_count":[1,2],
        "rebid_count":[0,1],
        "contractor_name":["Contractor A","Contractor B"],
        "source_url":["https://eprocure.gov.in/a","https://eprocure.gov.in/b"],
    }).to_csv(path,index=False)


def test_missing_verified_tender_file_returns_none(tmp_path:Path):
    assert load_verified_tenders(tmp_path/"missing.csv") is None


def test_tenders_require_source_attribution(tmp_path:Path):
    p=tmp_path/"t.csv";_write(p);x=pd.read_csv(p);x.loc[0,"source_url"]="";x.to_csv(p,index=False)
    with pytest.raises(ValueError):load_verified_tenders(p)


def test_future_award_cannot_change_earlier_snapshot(tmp_path:Path):
    p=tmp_path/"t.csv";_write(p);t=load_verified_tenders(p)
    score=pd.DataFrame({"canonical_project_id":["P1"],"snapshot_date":["2020-06-01"],"approved_cost_cr":[100]})
    out=attach_tender_economics(score,t)
    # Only the Jan-2020 award is visible: 90 vs 100 estimate = -10%.
    assert abs(out.loc[0,"exp_j_award_discount_pct"]+10.0)<1e-9
    assert out.loc[0,"exp_j_rebid_count"]==0
