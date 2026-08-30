"""Experiment 61: orthogonal composition of Exp51 Cost and Exp58 Delay.

Cost predictions come exclusively from Exp51's fold-stable shrunk residual
calibration. Delay predictions come exclusively from Exp58's normalized taxonomy
and strict temporal hierarchical-prior challenger. Component model choices and
hyperparameters are unchanged.
"""
import sys
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.nextgen_common import _persist, fit_cost_calibration, fit_priors, run_cli

EXPERIMENT_ID = "exp_61"
EXPERIMENT_SEQUENCE = 61
MARKER = "EXP61"
EXPERIMENT_NAME = "Exp51 shrunk Cost calibration + Exp58 hierarchical-prior Delay"
EXPERIMENT_SCOPE = "cost+delay"
CHANGED_DIMENSION = "exp51_cost_plus_exp58_delay"


def _ledger(path):
    frame = pd.read_csv(Path(path), parse_dates=["snapshot_date"])
    return frame.sort_values(["canonical_project_id", "snapshot_date"], kind="mergesort").reset_index(drop=True)


def fit_experiment(*, data, production_bundle, training_start, training_end, test_end, **kwargs):
    cost = fit_cost_calibration(
        data=data,
        production_bundle=production_bundle,
        training_start=training_start,
        training_end=training_end,
        test_end=test_end,
        exp_id=EXPERIMENT_ID,
        name=EXPERIMENT_NAME,
        strength=40.0,
    )
    prior = fit_priors(
        data=data,
        production_bundle=production_bundle,
        training_start=training_start,
        training_end=training_end,
        test_end=test_end,
        exp_id=EXPERIMENT_ID,
        name=EXPERIMENT_NAME,
    )

    c = _ledger(cost["experiment"]["ledger_path"])
    d = _ledger(prior["experiment"]["ledger_path"])
    keys = ["canonical_project_id", "snapshot_date"]
    if not c[keys].equals(d[keys]):
        raise AssertionError("Exp51 and Exp58 component ledgers do not share the exact cohort/order")

    required = [
        "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight",
        "production_cost_prediction", "production_delay_prediction",
    ]
    for col in required:
        if col not in c or col not in d:
            raise AssertionError(f"component ledger missing required column: {col}")
    if not c[required].equals(d[required]):
        raise AssertionError("component ledgers disagree on actuals, weights, or production predictions")

    return _persist(
        EXPERIMENT_ID,
        EXPERIMENT_NAME,
        EXPERIMENT_SCOPE,
        CHANGED_DIMENSION,
        c,
        c["production_cost_prediction"].to_numpy(float),
        c["experiment_cost_prediction"].to_numpy(float),
        c["production_delay_prediction"].to_numpy(float),
        d["experiment_delay_prediction"].to_numpy(float),
        {
            "composition": "Exp51 Cost predictions + Exp58 Delay predictions",
            "cost_component": "fold-stable shrunk Cost residual calibration; strength=40.0",
            "delay_component": "normalized taxonomy + strict earlier-completion-year hierarchical Delay priors",
            "interaction_tuning": False,
            "component_hyperparameters_changed": False,
        },
    )


if __name__ == "__main__":
    run_cli(sys.modules[__name__])
