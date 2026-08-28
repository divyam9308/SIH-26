"""Experiment 35: future cost-revision hurdle model (Cost only).

The challenger decomposes final Cost overrun into (1) whether the final overrun
will materially differ from the current as-of cost escalation and (2) the signed
magnitude of that remaining revision.  Both models use only the production Cost
feature contract.  Delay is reconstructed from the currently promoted Exp34
path-dependence + rolling-OOF ensemble and is not challenged.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES, _blend_predict, _fit_delay_family_models, _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows, enrich_supervised_for_production, target_feature_contract,
)

EXPERIMENT_ID = "exp_35"
EXPERIMENT_NAME = "Future cost-revision hurdle model"
EXPERIMENT_SCOPE = "cost"
EXPERIMENT_SEQUENCE = 35
COST_SEED = 26335
DELAY_SEED = 26204
REVISION_THRESHOLD_PP = 1.0


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _add_hurdle_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    current = pd.to_numeric(result.get("cost_escalation_percentage"), errors="coerce")
    final = pd.to_numeric(result["actual_cost_overrun_percentage"], errors="coerce")
    result["exp35_current_cost"] = current
    result["exp35_remaining_revision"] = final - current
    result["exp35_revision_event"] = result["exp35_remaining_revision"].abs().gt(REVISION_THRESHOLD_PP).astype(int)
    return result


def _current_delay_baseline(train: pd.DataFrame, delay_features: list[str]):
    weights, diagnostics = _oof_delay_weights(train, delay_features)
    models = _fit_delay_family_models(train, delay_features)
    return models, weights, diagnostics


def _fit_classifier(train: pd.DataFrame, features: list[str]):
    labels = train["exp35_revision_event"]
    if labels.nunique() < 2:
        return None, float(labels.mean())
    model = RandomForestClassifier(
        n_estimators=280, min_samples_leaf=4, class_weight="balanced_subsample",
        random_state=COST_SEED, n_jobs=2,
    )
    return _fit_pipeline(model, train, features, "exp35_revision_event"), None


def _event_probability(model, constant: float | None, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if model is None:
        return np.full(len(frame), float(constant or 0.0))
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(frame))
    return model.predict_proba(frame[features])[:, classes.index(1)]


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    contract = target_feature_contract(dict(production_bundle.get("metadata") or {}))
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    hurdle_train = _add_hurdle_targets(train).dropna(subset=["exp35_current_cost", "exp35_remaining_revision"])
    if hurdle_train["canonical_project_id"].nunique() < 10:
        raise ValueError("Experiment 35 has insufficient as-of cost-escalation history.")

    classifier, constant_probability = _fit_classifier(hurdle_train, cost_features)
    event_train = hurdle_train[hurdle_train["exp35_revision_event"].eq(1)].copy()
    if event_train["canonical_project_id"].nunique() < 5:
        raise ValueError("Experiment 35 has insufficient material cost-revision events.")
    family = str((production_bundle.get("metadata") or {}).get("selected_algorithms", {}).get("cost", "extra_trees"))
    magnitude_model = _fit_pipeline(_regressors(COST_SEED)[family], event_train, cost_features, "exp35_remaining_revision")
    lo, hi = np.nanpercentile(event_train["exp35_remaining_revision"].to_numpy(float), [1, 99])

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[cost_features])
    candidate_rows = _add_hurdle_targets(compare)
    probability = _event_probability(classifier, constant_probability, candidate_rows, cost_features)
    remaining = np.clip(magnitude_model.predict(candidate_rows[cost_features]), lo, hi)
    current = pd.to_numeric(candidate_rows["exp35_current_cost"], errors="coerce").to_numpy(float)
    current = np.where(np.isfinite(current), current, prod_cost_pred)
    exp_cost_pred = current + probability * remaining

    prod_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], prod_cost_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_cost = _regression_metrics(compare["actual_cost_overrun_percentage"], exp_cost_pred, compare["sample_weight"], compare["canonical_project_id"])

    delay_models, delay_weights, delay_oof = _current_delay_baseline(train, delay_features)
    prod_delay_pred = np.maximum(0, _blend_predict(delay_models, delay_weights, compare, delay_features))
    prod_delay = _regression_metrics(compare["actual_delay_days"], prod_delay_pred, compare["sample_weight"], compare["canonical_project_id"])
    exp_delay = dict(prod_delay)

    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    verdict = "PROMOTION CANDIDATE" if cost_gain > 0 else "REGRESSION / DO NOT PROMOTE"
    lookup = {
        _key(row): {
            "features": {name: row.get(name) for name in cost_features + delay_features},
            "current_cost": row.get("cost_escalation_percentage"),
        }
        for _, row in compare.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": f"exp35-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment", "promotion_allowed": False,
            "changed_dimension": "future_cost_revision_event_plus_signed_magnitude",
            "revision_threshold_pp": REVISION_THRESHOLD_PP,
            "cost_family": family, "future_holdout_used_for_selection": False,
            "delay_policy": "current_exp34_production_retained_exactly", "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": 0.0,
            "comparison_test_projects": int(compare["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "training_revision_event_rate": round(float(hurdle_train["exp35_revision_event"].mean()), 6),
            "delay_blend_weights": delay_weights, "delay_rolling_oof": delay_oof,
            "decision": verdict,
        },
        "runtime_state": {
            "production_cost_model": production_bundle["cost"], "cost_features": cost_features,
            "classifier": classifier, "constant_probability": constant_probability,
            "magnitude_model": magnitude_model, "magnitude_clip": (float(lo), float(hi)),
            "delay_models": delay_models, "delay_weights": delay_weights, "delay_features": delay_features,
            "lookup": lookup, "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 35 comparable snapshot is available.")
    candidate = row.copy()
    for name, value in state["lookup"][key]["features"].items():
        candidate[name] = value
    one = candidate.to_frame().T
    prod_cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    p = float(_event_probability(state["classifier"], state["constant_probability"], one, state["cost_features"])[0])
    remaining = float(state["magnitude_model"].predict(one[state["cost_features"]])[0])
    lo, hi = state["magnitude_clip"]
    remaining = float(np.clip(remaining, lo, hi))
    current = pd.to_numeric(pd.Series([state["lookup"][key]["current_cost"]]), errors="coerce").iloc[0]
    anchor = float(current) if pd.notna(current) else prod_cost
    cost = anchor + p * remaining
    delay = max(0.0, float(_blend_predict(state["delay_models"], state["delay_weights"], one, state["delay_features"])[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
