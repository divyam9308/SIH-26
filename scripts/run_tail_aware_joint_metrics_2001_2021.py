from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.nextgen_common import _prepare, normalize_taxonomy
from backend.app.ml.experiments.tail_aware_joint_metrics import (
    COST_FEATURES,
    DELAY_FEATURES,
    apply_tail_aware_layer,
    fit_tail_aware_layer,
    metrics,
)
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights, build_training_dataset
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
from backend.app.ml.production_exp35_baseline import CALIBRATION_GATE_FEATURE, _aft_routing_limit, _select_aft_calibration_projects
from backend.app.ml.production_exp61_baseline import _build_temporal_delay_priors
from backend.app.ml.production_exp105_exp113_baseline import (
    _current_cost_oof,
    _current_delay_oof,
    train_window_with_promoted_cost_and_delay,
)

TRAINING_START = 2001
TRAINING_END = 2021
TEST_END = 2025
OUT = Path("test-output/tail-aware-joint-metrics-2001-2021")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, identity = build_training_dataset()
    baseline_root = OUT / "baseline"
    result = train_window_with_promoted_cost_and_delay(
        TRAINING_START,
        TRAINING_END,
        TEST_END,
        data=data,
        identity=identity,
        artifact_root=baseline_root,
        verify_frozen_reference=True,
    )
    target = baseline_root / "2001_2021"
    cost_model = joblib.load(target / "cost_model.pkl")
    delay_model = joblib.load(target / "delay_model.pkl")

    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(prepared, TRAINING_START, TRAINING_END, TEST_END)
    prior_train, prior_test, _ = _build_temporal_delay_priors(train, test)
    cohort = _production_cost_evaluation_rows(prior_test).copy()
    calibration_ids = _select_aft_calibration_projects(
        cohort,
        limit=_aft_routing_limit(TRAINING_START, TRAINING_END, TEST_END),
    )
    cohort[CALIBRATION_GATE_FEATURE] = cohort["canonical_project_id"].astype("string").isin(calibration_ids)
    cohort = assign_project_balanced_weights(cohort)

    # Forward OOF evidence is generated only from the 2001-2021 training period.
    # The embedded base models are used so the new layer remains an isolated
    # challenger on top of the current Exp105/Exp113 production predictions.
    cost_oof = _current_cost_oof(prior_train, cost_model.base_model)
    delay_oof = _current_delay_oof(prior_train, delay_model.base_model)

    cost_layer = fit_tail_aware_layer(
        cost_oof,
        actual_col="actual_cost_overrun_percentage",
        features=COST_FEATURES,
        target="cost",
        seed=19501,
        nonnegative=False,
    )
    delay_layer = fit_tail_aware_layer(
        delay_oof,
        actual_col="actual_delay_days",
        features=DELAY_FEATURES,
        target="delay",
        seed=19502,
        nonnegative=True,
    )

    cost_anchor = np.asarray(cost_model.predict(cohort), dtype=float)
    delay_anchor = np.maximum(0.0, np.asarray(delay_model.predict(cohort), dtype=float))
    cost_candidate = apply_tail_aware_layer(cost_layer, cohort, cost_anchor, nonnegative=False)
    delay_candidate = apply_tail_aware_layer(delay_layer, cohort, delay_anchor, nonnegative=True)
    w = pd.to_numeric(cohort["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)

    baseline_cost = metrics(cohort["actual_cost_overrun_percentage"], cost_anchor, w)
    candidate_cost = metrics(cohort["actual_cost_overrun_percentage"], cost_candidate, w)
    baseline_delay = metrics(cohort["actual_delay_days"], delay_anchor, w)
    candidate_delay = metrics(cohort["actual_delay_days"], delay_candidate, w)

    def delta(candidate, baseline):
        return {k: float(candidate[k] - baseline[k]) for k in ("MAE", "RMSE", "R2")}

    cost_accept = (
        candidate_cost["MAE"] < baseline_cost["MAE"]
        and candidate_cost["RMSE"] < baseline_cost["RMSE"]
        and candidate_cost["R2"] > baseline_cost["R2"]
    )
    delay_accept = (
        candidate_delay["MAE"] < baseline_delay["MAE"]
        and candidate_delay["RMSE"] < baseline_delay["RMSE"]
        and candidate_delay["R2"] > baseline_delay["R2"]
    )

    report = {
        "experiment_id": "exp_195_tail_aware_joint_metrics",
        "model_role": "experiment",
        "window": {"training_start": 2001, "training_end": 2021, "test_start": 2022, "test_end": 2025},
        "selection_policy": "all correction hyperparameters and scales selected only from forward 2001-2021 OOF evidence",
        "holdout_used_for_selection": False,
        "full_holdout_retained": True,
        "production_result": result.get("lifecycle", {}).get("metrics", {}),
        "cost": {
            "baseline": baseline_cost,
            "candidate": candidate_cost,
            "delta_candidate_minus_baseline": delta(candidate_cost, baseline_cost),
            "tail_thresholds_training_oof": {"p90": cost_layer.p90, "p95": cost_layer.p95},
            "selected_scale": cost_layer.scale,
            "accept": bool(cost_accept),
        },
        "delay": {
            "baseline": baseline_delay,
            "candidate": candidate_delay,
            "delta_candidate_minus_baseline": delta(candidate_delay, baseline_delay),
            "tail_thresholds_training_oof": {"p90": delay_layer.p90, "p95": delay_layer.p95},
            "selected_scale": delay_layer.scale,
            "accept": bool(delay_accept),
        },
        "decision": "ACCEPT" if cost_accept and delay_accept else "REJECT",
        "promotion_allowed": False,
        "acceptance_contract": "Cost and Delay must each improve MAE, RMSE, and R2 on the full frozen holdout; otherwise reject.",
    }
    (OUT / "experiment_result.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
