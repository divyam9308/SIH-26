"""Experiment 33: cross-fitted median residual calibration."""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_33"
EXPERIMENT_NAME = "Cross-fitted median residual calibration"
EXPERIMENT_SCOPE = "cost+delay"
EXPERIMENT_SEQUENCE = 33
DELAY_SEED = 26204
MAX_FOLDS = 4
N_BINS = 5
MIN_GROUP_ROWS = 20


def _gain(b, c):
    return (b - c) / b * 100.0 if b else 0.0


def _key(r):
    return str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[mask], weights[mask]
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = 0.5 * weights.sum()
    if cutoff <= 0:
        return float(np.median(values))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _rolling_folds(train):
    years = sorted(
        int(y)
        for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique()
    )
    folds = []
    for year in reversed(years[1:]):
        completion_year = pd.to_numeric(train.completion_year, errors="coerce")
        fitting = train[completion_year.lt(year)].copy()
        validation = train[completion_year.eq(year)].copy()
        if (
            fitting.canonical_project_id.nunique() >= 10
            and validation.canonical_project_id.nunique() >= 3
        ):
            folds.append((fitting, validation, year))
        if len(folds) >= MAX_FOLDS:
            break
    return list(reversed(folds))


def _fit_calibration(train, features, target, algorithm, seed):
    folds = _rolling_folds(train)
    if len(folds) < 2:
        raise ValueError("Experiment 33 requires at least two rolling OOF folds.")

    chunks = []
    for fitting, validation, year in folds:
        model = _fit_pipeline(_regressors(seed)[algorithm], fitting, features, target)
        pred = model.predict(validation[features])
        pred = np.maximum(0, pred) if target == "actual_delay_days" else pred
        chunk = validation[
            [target, "sample_weight", "canonical_project_id", "lifecycle_stage"]
        ].copy()
        chunk["prediction"] = pred
        chunk["residual"] = pd.to_numeric(chunk[target], errors="coerce") - pred
        chunk["validation_year"] = year
        chunks.append(chunk)

    oof = pd.concat(chunks, ignore_index=True)
    finite = np.isfinite(oof.prediction.to_numpy(float))
    if finite.sum() < 20:
        raise ValueError("Experiment 33 has insufficient finite OOF predictions.")

    edges = np.unique(
        np.quantile(
            oof.loc[finite, "prediction"], np.linspace(0, 1, N_BINS + 1)
        ).astype(float)
    )
    if len(edges) < 3:
        edges = np.array([-np.inf, float(np.median(oof.prediction)), np.inf])
    else:
        # Unbounded edge sentinels are required by the runtime binning logic.
        # They remain in runtime_state only; public diagnostics are sanitized
        # before strict JSON serialization.
        edges[0] = -np.inf
        edges[-1] = np.inf

    oof["bin"] = np.digitize(
        oof.prediction.to_numpy(float), edges[1:-1], right=False
    )
    global_median = _weighted_median(oof.residual, oof.sample_weight)
    bin_medians = {
        int(b): _weighted_median(part.residual, part.sample_weight)
        for b, part in oof.groupby("bin")
    }
    stage_bin = {}
    for (stage, b), part in oof.groupby(
        ["lifecycle_stage", "bin"], dropna=False
    ):
        if len(part) < MIN_GROUP_ROWS:
            continue
        stage_key = "<NA>" if pd.isna(stage) else str(stage)
        stage_bin[(stage_key, int(b))] = _weighted_median(
            part.residual, part.sample_weight
        )

    return {
        "edges": edges.tolist(),
        "global_median": global_median,
        "bin_medians": bin_medians,
        "stage_bin_medians": stage_bin,
        "oof_rows": int(len(oof)),
        "fold_years": [int(y) for _, _, y in folds],
    }


def _public_calibration(calibration):
    """Return strict-JSON-safe calibration diagnostics.

    Runtime calibration intentionally uses -inf/+inf as open-ended bin edges.
    JSON artifacts use ``allow_nan=False`` by design, so those sentinels must not
    escape into the public experiment payload. ``None`` explicitly represents
    an unbounded diagnostic edge while runtime_state retains the numeric edges.
    """
    edges = []
    for value in calibration["edges"]:
        numeric = float(value)
        edges.append(numeric if np.isfinite(numeric) else None)
    return {
        "edges": edges,
        "edge_semantics": "null first/last edge means unbounded",
        "global_median": float(calibration["global_median"]),
        "oof_rows": int(calibration["oof_rows"]),
        "fold_years": [int(year) for year in calibration["fold_years"]],
    }


def _corrections(frame, predictions, calibration):
    edges = np.asarray(calibration["edges"], dtype=float)
    bins = np.digitize(
        np.asarray(predictions, dtype=float), edges[1:-1], right=False
    )
    result = np.zeros(len(frame))
    stages = frame.get(
        "lifecycle_stage", pd.Series(pd.NA, index=frame.index)
    )
    for i, (stage, b) in enumerate(zip(stages, bins)):
        stage_key = "<NA>" if pd.isna(stage) else str(stage)
        corr = calibration["stage_bin_medians"].get((stage_key, int(b)))
        if corr is None:
            corr = calibration["bin_medians"].get(
                int(b), calibration["global_median"]
            )
        result[i] = float(corr)
    return result


def fit_experiment(
    *,
    data,
    training_start,
    training_end,
    test_end,
    production_bundle,
    production_receipt,
    **_,
):
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(
        enriched.completion_year, errors="coerce"
    )
    enriched["snapshot_date"] = pd.to_datetime(
        enriched.snapshot_date, errors="coerce"
    )
    train, test = temporal_project_split(
        enriched, training_start, training_end, test_end
    )

    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    selected = dict(
        metadata.get("selected_algorithms")
        or production_receipt.get("selected_algorithms")
        or {}
    )
    cost_name, delay_name = selected.get("cost"), selected.get("delay")
    cost_features, delay_features = list(contract["cost"]), list(contract["delay"])

    cost_cal = _fit_calibration(
        train,
        cost_features,
        "actual_cost_overrun_percentage",
        cost_name,
        PRODUCTION_COST_SEED,
    )
    delay_cal = _fit_calibration(
        train,
        delay_features,
        "actual_delay_days",
        delay_name,
        DELAY_SEED,
    )

    cost_compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(cost_compare[cost_features])
    exp_cost_pred = prod_cost_pred + _corrections(
        cost_compare, prod_cost_pred, cost_cal
    )
    prod_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage,
        prod_cost_pred,
        cost_compare.sample_weight,
        cost_compare.canonical_project_id,
    )
    exp_cost = _regression_metrics(
        cost_compare.actual_cost_overrun_percentage,
        exp_cost_pred,
        cost_compare.sample_weight,
        cost_compare.canonical_project_id,
    )

    prod_delay_pred = np.maximum(
        0, production_bundle["delay"].predict(test[delay_features])
    )
    exp_delay_pred = np.maximum(
        0, prod_delay_pred + _corrections(test, prod_delay_pred, delay_cal)
    )
    prod_delay = _regression_metrics(
        test.actual_delay_days,
        prod_delay_pred,
        test.sample_weight,
        test.canonical_project_id,
    )
    exp_delay = _regression_metrics(
        test.actual_delay_days,
        exp_delay_pred,
        test.sample_weight,
        test.canonical_project_id,
    )

    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    verdict = (
        "PROMOTION CANDIDATE"
        if cost_gain >= 0
        and delay_gain >= 0
        and (cost_gain > 0 or delay_gain > 0)
        else "REGRESSION / DO NOT PROMOTE"
    )

    union = list(dict.fromkeys(cost_features + delay_features))
    lookup = {
        _key(row): {name: row.get(name) for name in union + ["lifecycle_stage"]}
        for _, row in test.iterrows()
    }

    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": (
                f"exp33-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
            ),
            "model_role": "experiment",
            "promotion_allowed": False,
            "changed_dimension": "cross_fitted_post_model_calibration",
            "selected_algorithms": selected,
            "cost_calibration": _public_calibration(cost_cal),
            "delay_calibration": _public_calibration(delay_cal),
            "future_holdout_used_for_calibration": False,
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(test.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(test)),
            "cost_comparison_projects": int(
                cost_compare.canonical_project_id.nunique()
            ),
            "decision": verdict,
        },
        "runtime_state": {
            "cost_model": production_bundle["cost"],
            "delay_model": production_bundle["delay"],
            "cost_features": cost_features,
            "delay_features": delay_features,
            "cost_calibration": cost_cal,
            "delay_calibration": delay_cal,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame, state):
    return frame[
        frame.apply(lambda row: _key(row) in state["comparable"], axis=1)
    ].copy()


def predict_project(row, state):
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 33 feature vector is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    production_cost = float(
        state["cost_model"].predict(one.reindex(columns=state["cost_features"]))[0]
    )
    production_delay = max(
        0.0,
        float(
            state["delay_model"].predict(
                one.reindex(columns=state["delay_features"])
            )[0]
        ),
    )
    cost_correction = float(
        _corrections(
            one, np.array([production_cost]), state["cost_calibration"]
        )[0]
    )
    delay_correction = float(
        _corrections(
            one, np.array([production_delay]), state["delay_calibration"]
        )[0]
    )
    return {
        "predicted_cost_overrun": round(production_cost + cost_correction, 4),
        "predicted_delay_days": round(
            max(0.0, production_delay + delay_correction), 4
        ),
        "cost_median_residual_correction": round(cost_correction, 4),
        "delay_median_residual_correction": round(delay_correction, 4),
    }
