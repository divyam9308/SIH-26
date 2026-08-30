"""Experiment 66: training-only hierarchical residual correction for Exp61."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from backend.app.ml.experiments.nextgen_common import _persist
from backend.app.ml.experiments.post61_common import cost_oof_frame, delay_oof_frame, production_comparison, run_cli, weighted_quantile

EXPERIMENT_ID = "exp_66"
EXPERIMENT_SEQUENCE = 66
MARKER = "EXP66"
EXPERIMENT_NAME = "U5 hierarchical residual shrinkage on Exp61"
EXPERIMENT_SCOPE = "cost+delay"
CHANGED_DIMENSION = "hierarchical_context_residual_correction"
STRENGTH = 25.0
LEVELS = [
    ("lifecycle_stage",),
    ("lifecycle_stage", "project_size_category"),
    ("_norm_sector", "lifecycle_stage", "project_size_category"),
    ("_norm_implementing_agency", "_norm_sector", "lifecycle_stage", "project_size_category"),
]


def _weighted_median(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v, w = v[good], w[good]
    if not len(v):
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    total = float(w.sum())
    if total <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * total, side="left")])


def _normalize_key(values):
    return tuple("<NA>" if pd.isna(x) else str(x) for x in values)


def _fit_hierarchy(oof: pd.DataFrame, score: pd.DataFrame):
    residual = pd.to_numeric(oof["residual"], errors="coerce").fillna(0.0)
    weight = pd.to_numeric(oof["sample_weight"], errors="coerce").fillna(0.0)
    global_value = _weighted_median(residual, weight)
    maps = []
    parent = {(): global_value}
    parent_len = 0
    for keys in LEVELS:
        current = {}
        for raw_key, part in oof.groupby(list(keys), dropna=False):
            if not isinstance(raw_key, tuple):
                raw_key = (raw_key,)
            key = _normalize_key(raw_key)
            support = float(pd.to_numeric(part["sample_weight"], errors="coerce").fillna(0.0).sum())
            local = _weighted_median(part["residual"], part["sample_weight"])
            parent_value = parent.get(key[:parent_len] if parent_len else (), global_value)
            alpha = support / (support + STRENGTH)
            current[key] = (alpha * local + (1.0 - alpha) * parent_value, support)
        maps.append((keys, current))
        parent = {key: value[0] for key, value in current.items()}
        parent_len = len(keys)

    correction = np.full(len(score), global_value, dtype=float)
    deepest = np.zeros(len(score), dtype=int)
    for depth, (keys, mapping) in enumerate(maps, start=1):
        for pos, (_, row) in enumerate(score.iterrows()):
            key = tuple("<NA>" if pd.isna(row.get(k)) else str(row.get(k)) for k in keys)
            if key in mapping:
                correction[pos] = mapping[key][0]
                deepest[pos] = depth
    cap = weighted_quantile(np.abs(residual), weight, 0.90)
    correction = np.clip(correction, -cap, cap)
    return correction, {
        "strength": STRENGTH,
        "global_residual": global_value,
        "correction_cap_q90": cap,
        "most_specific_rows": int((deepest == len(LEVELS)).sum()),
        "holdout_tuned": False,
    }


def fit_experiment(*, data, production_bundle, training_start, training_end, test_end, **kwargs):
    _, _, cohort, production_cost, production_delay = production_comparison(data, production_bundle, training_start, training_end, test_end)
    cost_oof = cost_oof_frame(data, production_bundle, training_start, training_end, test_end)
    cost_corr, cost_details = _fit_hierarchy(cost_oof, cohort)
    delay_oof = delay_oof_frame(data, production_bundle, training_start, training_end, test_end)
    delay_corr, delay_details = _fit_hierarchy(delay_oof, cohort)
    return _persist(
        EXPERIMENT_ID, EXPERIMENT_NAME, EXPERIMENT_SCOPE, CHANGED_DIMENSION,
        cohort, production_cost, production_cost + cost_corr,
        production_delay, np.maximum(0.0, production_delay + delay_corr),
        {"baseline": "assumed Exp61 production from PR #96", "cost": cost_details, "delay": delay_details},
    )


if __name__ == "__main__":
    run_cli(sys.modules[__name__])
