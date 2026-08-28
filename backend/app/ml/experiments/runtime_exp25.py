"""Retrain & Compare runtime for the full Experiment 25 context challenger."""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd

from backend.app.ml.experiments.milestone_delay_exp25 import (
    ALL_ADDED_FEATURES,
    CONTEXT_FEATURES,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    MILESTONE_FEATURES,
    decision,
    enrich_exp25_features,
)
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    enrich_supervised_for_production,
    target_feature_contract,
)

FEATURE_GROUPS = {
    "production_contract": [],
    "project_semantics_context": CONTEXT_FEATURES,
    "milestone_trajectory": MILESTONE_FEATURES,
    "full_context_plus_milestones": ALL_ADDED_FEATURES,
}


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _available(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return list(dict.fromkeys(name for name in features if name in frame.columns))


def _selection_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    years = sorted(int(y) for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique())
    if len(years) < 3:
        return None
    for count in (2, 1):
        validation_years = set(years[-count:])
        fitting = train[~train.completion_year.isin(validation_years)]
        validation = train[train.completion_year.isin(validation_years)]
        if fitting.canonical_project_id.nunique() >= 10 and validation.canonical_project_id.nunique() >= 5:
            return fitting, validation
    return None


def _select_feature_group(
    train: pd.DataFrame,
    base_features: list[str],
    target: str,
    algorithm: str,
    seed: int,
) -> tuple[str, list[str], list[dict]]:
    """Choose an added feature group on training-period data only."""
    split = _selection_split(train)
    if split is None:
        return "production_contract", list(base_features), [{
            "group": "production_contract",
            "selected": True,
            "reason": "insufficient projects for an internal forward temporal feature-group validation",
        }]
    fitting, validation = split
    comparisons: list[dict] = []
    for group, additions in FEATURE_GROUPS.items():
        features = _available(train, list(base_features) + list(additions))
        model = _fit_pipeline(_regressors(seed)[algorithm], fitting, features, target)
        pred = model.predict(validation[features])
        if target == "actual_delay_days":
            pred = np.maximum(0, pred)
        metrics = _regression_metrics(
            validation[target], pred, validation.sample_weight, validation.canonical_project_id
        )
        comparisons.append({
            "group": group,
            "MAE": metrics["MAE"],
            "features": features,
            "added_features": [name for name in features if name not in base_features],
            "validation_projects": metrics["unique_projects"],
            "validation_rows": metrics["rows"],
        })
    winner = min(comparisons, key=lambda item: item["MAE"])
    for item in comparisons:
        item["selected"] = item["group"] == winner["group"]
    return str(winner["group"]), list(winner["features"]), comparisons


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_exp25_features(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    metadata = production_bundle.get("metadata") or {}
    contract = target_feature_contract(metadata)
    selected = dict(metadata.get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    if not selected.get("cost") or not selected.get("delay"):
        raise ValueError("Experiment 25 requires production-selected cost and delay algorithms.")

    cost_group, cost_features, cost_selection = _select_feature_group(
        train,
        list(contract["cost"]),
        "actual_cost_overrun_percentage",
        selected["cost"],
        PRODUCTION_COST_SEED,
    )
    delay_group, delay_features, delay_selection = _select_feature_group(
        train,
        list(contract["delay"]),
        "actual_delay_days",
        selected["delay"],
        26204,
    )

    cost_retained = cost_group == "production_contract"
    delay_retained = delay_group == "production_contract"
    cost_model = (
        production_bundle["cost"]
        if cost_retained
        else _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[selected["cost"]],
            train,
            cost_features,
            "actual_cost_overrun_percentage",
        )
    )
    delay_model = (
        production_bundle["delay"]
        if delay_retained
        else _fit_pipeline(
            _regressors(26204)[selected["delay"]],
            train,
            delay_features,
            "actual_delay_days",
        )
    )

    prod_cost_pred = production_bundle["cost"].predict(test[contract["cost"]])
    prod_delay_pred = np.maximum(0, production_bundle["delay"].predict(test[contract["delay"]]))
    exp_cost_pred = cost_model.predict(test[cost_features])
    exp_delay_pred = np.maximum(0, delay_model.predict(test[delay_features]))

    prod_cost = _regression_metrics(
        test.actual_cost_overrun_percentage,
        prod_cost_pred,
        test.sample_weight,
        test.canonical_project_id,
    )
    exp_cost = _regression_metrics(
        test.actual_cost_overrun_percentage,
        exp_cost_pred,
        test.sample_weight,
        test.canonical_project_id,
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
    verdict = decision(cost_gain, delay_gain)

    lookup = {
        _key(row): {feature: row.get(feature) for feature in ALL_ADDED_FEATURES}
        for _, row in test.iterrows()
    }
    run_id = f"exp25-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    train_types = train.exp25_project_type.astype("string")
    test_types = test.exp25_project_type.astype("string")
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": run_id,
            "model_role": "experiment",
            "promotion_allowed": False,
            "raw_project_name_used_as_feature": False,
            "candidate_feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
            "selected_feature_groups": {"cost": cost_group, "delay": delay_group},
            "selected_added_features": {
                "cost": [name for name in cost_features if name not in contract["cost"]],
                "delay": [name for name in delay_features if name not in contract["delay"]],
            },
            "selected_algorithms": selected,
            "internal_feature_group_comparisons": {
                "cost": cost_selection,
                "delay": delay_selection,
            },
            "metrics": {"cost": exp_cost, "delay": exp_delay},
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(test.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(test)),
            "selected_cost_feature_group": cost_group,
            "selected_delay_feature_group": delay_group,
            "training_milestone_snapshot_share": float(train.exp25_milestone_ratio.notna().mean()),
            "test_milestone_snapshot_share": float(test.exp25_milestone_ratio.notna().mean()),
            "training_typed_project_share": float(train_types.ne("other").mean()),
            "test_typed_project_share": float(test_types.ne("other").mean()),
            "training_state_share": float(train.state.notna().mean()) if "state" in train else 0.0,
            "test_state_share": float(test.state.notna().mean()) if "state" in test else 0.0,
            "feature_history_granularity": "full official monthly history",
            "decision": verdict,
        },
        "runtime_state": {
            "cost_model": cost_model,
            "delay_model": delay_model,
            "cost_features": cost_features,
            "delay_features": delay_features,
            "cost_group": cost_group,
            "delay_group": delay_group,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 25 context representation is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    cost_x = candidate.to_frame().T.reindex(columns=state["cost_features"])
    delay_x = candidate.to_frame().T.reindex(columns=state["delay_features"])
    return {
        "predicted_cost_overrun": round(float(state["cost_model"].predict(cost_x)[0]), 4),
        "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(delay_x)[0])), 4),
        "cost_feature_group": state["cost_group"],
        "delay_feature_group": state["delay_group"],
        "milestone_features_available": int(
            sum(pd.notna(candidate.get(feature)) for feature in MILESTONE_FEATURES)
        ),
    }
