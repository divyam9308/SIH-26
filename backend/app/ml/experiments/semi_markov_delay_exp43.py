"""Experiment 43: semi-Markov lifecycle transition model (Delay only).

Observable as-of lifecycle-stage x execution-stress states are learned from
training trajectories. Weighted state-to-state transition probabilities and
transition dwell times define a semi-Markov process with completion absorbing.
Expected remaining time to completion is solved from the training-only process
and converted to final Delay. Cost remains production Exp12 unchanged.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_43"
EXPERIMENT_NAME = "Semi-Markov lifecycle transition Delay model"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 43
COMPLETE = "COMPLETE"


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def observable_state(frame: pd.DataFrame) -> pd.Series:
    stage = frame.get("lifecycle_stage", pd.Series("unknown", index=frame.index)).astype("string").fillna("unknown")
    slip = pd.to_numeric(frame.get("schedule_slippage_days"), errors="coerce").fillna(0.0)
    progress = pd.to_numeric(frame.get("progress_deviation"), errors="coerce").fillna(0.0)
    healthy = slip.le(30) & progress.ge(-15)
    severe = slip.gt(365) | progress.lt(-30)
    stress = np.where(healthy, "healthy", np.where(severe, "severe", "stressed"))
    return stage.astype(str) + "|" + pd.Series(stress, index=frame.index).astype(str)


def fit_semi_markov(train: pd.DataFrame):
    work = train.copy()
    work["exp43_state"] = observable_state(work)
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
    work["completion_date"] = pd.to_datetime(work["completion_date"], errors="coerce")
    transitions = []
    remaining_by_state: dict[str, list[float]] = {}
    for _, group in work.sort_values(["canonical_project_id", "snapshot_date"]).groupby("canonical_project_id", sort=False):
        group = group.sort_values("snapshot_date")
        n_transitions = max(1, len(group))
        project_weight = 1.0 / n_transitions
        rows = list(group.itertuples(index=False))
        for i, row in enumerate(rows):
            state = str(row.exp43_state)
            remaining = (pd.Timestamp(row.completion_date) - pd.Timestamp(row.snapshot_date)).days
            if remaining > 0:
                remaining_by_state.setdefault(state, []).append(float(remaining))
            if i + 1 < len(rows):
                nxt = rows[i + 1]
                to_state = str(nxt.exp43_state)
                dwell = max(1.0, float((pd.Timestamp(nxt.snapshot_date) - pd.Timestamp(row.snapshot_date)).days))
            else:
                to_state = COMPLETE
                dwell = max(1.0, float((pd.Timestamp(row.completion_date) - pd.Timestamp(row.snapshot_date)).days))
            transitions.append((state, to_state, dwell, project_weight))
    states = sorted({item[0] for item in transitions} | {item[1] for item in transitions if item[1] != COMPLETE})
    if not states:
        raise ValueError("Experiment 43 found no observable lifecycle states.")
    index = {state: i for i, state in enumerate(states)}
    count = np.zeros((len(states), len(states) + 1), dtype=float)
    dwell_sum = np.zeros_like(count)
    for source, target, dwell, weight in transitions:
        i = index[source]
        j = len(states) if target == COMPLETE else index[target]
        count[i, j] += weight
        dwell_sum[i, j] += weight * dwell
    probabilities = np.zeros_like(count)
    mean_dwell = np.zeros_like(count)
    for i in range(len(states)):
        total = count[i].sum()
        if total <= 0:
            probabilities[i, -1] = 1.0
            mean_dwell[i, -1] = float(np.median(remaining_by_state.get(states[i], [365.0])))
            continue
        probabilities[i] = count[i] / total
        mask = count[i] > 0
        mean_dwell[i, mask] = dwell_sum[i, mask] / count[i, mask]
    q = probabilities[:, :len(states)]
    expected_step = (probabilities * mean_dwell).sum(axis=1)
    a = np.eye(len(states)) - q
    expected = np.linalg.lstsq(a, expected_step, rcond=None)[0]
    fallback = {state: float(np.median(values)) for state, values in remaining_by_state.items() if values}
    cap_values = [value for values in remaining_by_state.values() for value in values]
    cap = float(np.percentile(cap_values, 99)) if cap_values else 3650.0
    remaining = {}
    for state, value in zip(states, expected):
        if not np.isfinite(value) or value <= 0:
            value = fallback.get(state, 365.0)
        remaining[state] = float(np.clip(value, 1.0, max(365.0, cap)))
    global_fallback = float(np.median(cap_values)) if cap_values else 365.0
    return remaining, global_fallback, {
        "states": states,
        "transition_count": len(transitions),
        "state_count": len(states),
    }


def _delay_from_states(frame: pd.DataFrame, expected: dict[str, float], fallback: float) -> np.ndarray:
    states = observable_state(frame)
    days = np.asarray([expected.get(str(state), fallback) for state in states], dtype=float)
    snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    planned = pd.to_datetime(frame["planned_completion_date"], errors="coerce")
    completion = snapshot + pd.to_timedelta(days, unit="D")
    return np.maximum(0.0, (completion - planned).dt.days.to_numpy(float))


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    expected, fallback, diagnostics = fit_semi_markov(train)
    compare = _production_cost_evaluation_rows(test).copy()
    compare = compare[compare["planned_completion_date"].notna()].copy()
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    prod_weights, prod_oof = _oof_delay_weights(train, delay_features)
    prod_models = _fit_delay_family_models(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(prod_models, prod_weights, compare, delay_features))
    exp_delay_pred = _delay_from_states(compare, expected, fallback)
    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_delay = _regression_metrics(compare["actual_delay_days"], exp_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = "PROMOTION CANDIDATE" if gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup_features = list(dict.fromkeys(cost_features + delay_features + ["planned_completion_date", "lifecycle_stage", "progress_deviation", "schedule_slippage_days"]))
    lookup = {_key(row): {name: row.get(name) for name in lookup_features} for _, row in compare.iterrows()}
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp43-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "semi_markov_lifecycle_transition_process",
            "semi_markov_diagnostics": diagnostics, "future_holdout_used_for_selection": False,
            "cost_policy": "production_exp12_retained_exactly", "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": prod_cost["MAE"],
            "cost_improvement_percentage": 0.0,
            "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(gain, 4),
            "comparison_test_projects": int(compare["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(compare)), "production_delay_blend_weights": prod_weights,
            "production_delay_rolling_oof": prod_oof, "decision": verdict,
        },
        "runtime_state": {
            "production_cost_model": production_bundle["cost"], "cost_features": cost_features,
            "expected_remaining": expected, "fallback_remaining": fallback,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 43 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    delay = float(_delay_from_states(one, state["expected_remaining"], state["fallback_remaining"])[0])
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
