import numpy as np
import pandas as pd

from backend.app.ml.experiments.experiment_i_cost_budget_credibility import (
    PRIOR_FEATURES,
    attach_budget_credibility,
    crossfit_training_priors,
)


def _data():
    rows=[]
    for year in range(2015,2022):
        for i in range(12):
            approved=100.0; revised=110.0+(i%3)*5; final=revised+10+(year-2015)
            rows.append({"canonical_project_id":f"{year}-{i}","completion_year":year,"approved_cost_cr":approved,"revised_cost_cr":revised,"reported_completion_expenditure_cr":final,"cost_escalation_percentage":revised/approved*100-100,"implementing_agency":"Agency A" if i<8 else "Agency B","sector":"Roads" if i%2 else "Railways","actual_cost_overrun_percentage":99999+i})
    return pd.DataFrame(rows)


def test_holdout_cost_label_cannot_change_budget_prior():
    train=_data();score=train.iloc[:5].copy();first=attach_budget_credibility(train,score);score["actual_cost_overrun_percentage"]=[-999999]*len(score);second=attach_budget_credibility(train,score)
    for feature in PRIOR_FEATURES:
        assert np.allclose(pd.to_numeric(first[feature],errors="coerce").fillna(-9999),pd.to_numeric(second[feature],errors="coerce").fillna(-9999))


def test_crossfit_uses_no_same_or_future_completion_year_for_earliest_rows():
    train=_data();cross=crossfit_training_priors(train);earliest=cross.loc[cross["completion_year"]==train["completion_year"].min()]
    assert (earliest["exp_i_budget_bias_support"]==0).all()
    assert (earliest["exp_i_budget_bias_pct"]==0).all()
