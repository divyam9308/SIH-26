import numpy as np
import pandas as pd

from backend.app.ml.experiments.experiment_g_cost_learned_earned_cost import (
    EARNED_FEATURES,
    attach_earned_cost,
)


def _train():
    rows=[]
    for i in range(80):
        progress=float((i%20)*5+5)
        rows.append({
            "canonical_project_id":f"P{i}",
            "physical_progress":progress,
            "expenditure_ratio":0.15 + (progress/100.0)**1.4,
            "sector":"Roads" if i%2 else "Railways",
            "actual_cost_overrun_percentage":i*1000,
        })
    return pd.DataFrame(rows)


def test_earned_curve_does_not_use_cost_outcome():
    train=_train(); score=train.iloc[:10].copy()
    first=attach_earned_cost(train,score)
    changed=train.copy(); changed["actual_cost_overrun_percentage"]=np.arange(len(changed))*999999
    second=attach_earned_cost(changed,score)
    for feature in EARNED_FEATURES:
        assert np.allclose(
            pd.to_numeric(first[feature],errors="coerce").fillna(-9999),
            pd.to_numeric(second[feature],errors="coerce").fillna(-9999),
        )


def test_spend_above_learned_curve_has_positive_gap():
    train=_train()
    score=pd.DataFrame({"physical_progress":[50.0],"expenditure_ratio":[2.0],"sector":["Roads"]})
    out=attach_earned_cost(train,score)
    assert out.loc[0,"exp_g_spend_vs_norm"] > 0
