"""Experiment 25 retest on the current Exp12 Cost + Exp34 Delay production stack.

The challenger adds project-name-derived reusable semantics, structured PAIMANA
context, and causal milestone trajectory features. Raw project names are never
passed to a model. Feature-group choice is made only inside the training window.

Cost holds the current production-selected family fixed. Delay holds the current
Exp34 blend architecture and OOF-selected weights fixed, so this retest isolates
whether Exp25 context adds signal on top of the promoted production stack.
"""
from __future__ import annotations

import re
import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.path_oof_delay_exp34 import (
    FAMILIES,
    _blend_predict,
    _fit_delay_family_models,
    enrich_path_dependence,
)
from backend.app.ml.monthly_lifecycle import TRAJECTORIES
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

EXPERIMENT_ID = "exp_25_current"
EXPERIMENT_NAME = "Exp25 context + milestones on current Exp12/Exp34 production"
EXPERIMENT_SCOPE = "cost+delay"
EXPERIMENT_SEQUENCE = 36
DELAY_SEED = 26204

PROJECT_TYPE_RULES = [
    ("metro", r"\bmetro\b"),
    ("railway", r"\b(rail|railway|rail line|doubling|gauge conversion)\b"),
    ("airport", r"\b(airport|aerodrome|runway)\b"),
    ("port", r"\b(port|harbour|harbor|jetty|berth)\b"),
    ("hydro_power", r"\b(hydro|hydel|hydroelectric)\b"),
    ("thermal_power", r"\b(thermal|supercritical|ultra supercritical)\b"),
    ("solar_power", r"\bsolar\b"),
    ("wind_power", r"\bwind\b"),
    ("power_grid", r"\b(transmission|substation|power grid|power evacuation)\b"),
    ("bridge", r"\b(bridge|flyover|viaduct)\b"),
    ("tunnel", r"\btunnel\b"),
    ("road", r"\b(highway|expressway|road|bypass|ring road|lane)\b"),
    ("irrigation_water", r"\b(irrigation|canal|dam|reservoir|water supply|sewer|sewage)\b"),
    ("pipeline", r"\b(pipeline|pipe line|gas line|oil line)\b"),
    ("hospital", r"\b(hospital|medical college|medical institute)\b"),
    ("building", r"\b(building|housing|campus|complex|office)\b"),
]

SEMANTIC_FEATURES = [
    "exp25_project_type",
    "exp25_has_phase",
    "exp25_has_stage",
    "exp25_has_package",
    "exp25_has_corridor",
    "exp25_has_extension",
    "exp25_has_greenfield",
    "exp25_has_modernisation",
    "exp25_phase_number",
    "exp25_stage_number",
    "exp25_package_number",
    "exp25_lane_count",
    "exp25_capacity_mw",
    "exp25_length_km",
    "exp25_unit_count",
    "exp25_scope_marker_count",
    "exp25_sector_project_type",
    "exp25_state_project_type",
]

STRUCTURED_CONTEXT_FEATURES = [
    "sector",
    "ministry",
    "implementing_agency",
    "state",
    "approved_cost_cr",
    "revised_cost_cr",
    "cumulative_expenditure_cr",
    "physical_progress",
    "current_schedule_status",
    "exp25_financial_progress",
    "exp25_financial_physical_gap",
]

MILESTONE_FEATURES = [
    "exp25_milestones_achieved",
    "exp25_milestones_total",
    "exp25_milestone_ratio",
    "exp25_milestones_remaining",
    "exp25_milestone_velocity",
    "exp25_milestone_delta",
    "exp25_milestone_stagnant",
    "exp25_months_since_milestone_change",
]

CONTEXT_FEATURES = list(dict.fromkeys(SEMANTIC_FEATURES + STRUCTURED_CONTEXT_FEATURES))
ALL_ADDED_FEATURES = list(dict.fromkeys(CONTEXT_FEATURES + MILESTONE_FEATURES))
FEATURE_GROUPS = {
    "production_contract": [],
    "project_semantics_context": CONTEXT_FEATURES,
    "milestone_trajectory": MILESTONE_FEATURES,
    "full_context_plus_milestones": ALL_ADDED_FEATURES,
}


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip().lower()


def _number_from(text: str, pattern: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return np.nan


def _ordinal_from(text: str, word: str) -> float:
    numeric = _number_from(text, rf"\b{word}\s*[-:/]?\s*(\d{{1,3}})\b")
    if pd.notna(numeric):
        return numeric
    roman = re.search(
        rf"\b{word}\s*[-:/]?\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not roman:
        return np.nan
    mapping = {
        "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
        "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
    }
    return float(mapping[roman.group(1).lower()])


def _project_type(text: str) -> str:
    for label, pattern in PROJECT_TYPE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "other"


def add_project_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    names = result.get("project_name", pd.Series("", index=result.index)).map(_text)
    sectors = result.get("sector", pd.Series("", index=result.index)).map(_text)
    states = result.get("state", pd.Series("", index=result.index)).map(_text)

    result["exp25_project_type"] = names.map(_project_type).astype("string")
    flag_patterns = {
        "exp25_has_phase": r"\bphase\b",
        "exp25_has_stage": r"\bstage\b",
        "exp25_has_package": r"\b(package|pkg)\b",
        "exp25_has_corridor": r"\bcorridor\b",
        "exp25_has_extension": r"\b(extension|extn)\b",
        "exp25_has_greenfield": r"\bgreenfield\b",
        "exp25_has_modernisation": r"\b(modernisation|modernization|upgradation|upgrade)\b",
    }
    for feature, pattern in flag_patterns.items():
        result[feature] = names.str.contains(pattern, case=False, regex=True, na=False).astype(float)

    result["exp25_phase_number"] = names.map(lambda value: _ordinal_from(value, "phase"))
    result["exp25_stage_number"] = names.map(lambda value: _ordinal_from(value, "stage"))
    result["exp25_package_number"] = names.map(
        lambda value: _number_from(value, r"\b(?:package|pkg)\s*[-:/]?\s*(\d{1,3})\b")
    )
    result["exp25_lane_count"] = names.map(
        lambda value: _number_from(value, r"\b(\d{1,2})\s*[- ]?lane\b")
    )
    result["exp25_capacity_mw"] = names.map(
        lambda value: _number_from(value, r"\b(\d+(?:\.\d+)?)\s*mw\b")
    )
    result["exp25_length_km"] = names.map(
        lambda value: _number_from(value, r"\b(\d+(?:\.\d+)?)\s*km\b")
    )
    result["exp25_unit_count"] = names.map(
        lambda value: _number_from(value, r"\b(\d{1,3})\s*(?:units?|nos?\.?\s*units?)\b")
    )
    result["exp25_scope_marker_count"] = result[list(flag_patterns)].sum(axis=1)

    ptype = result["exp25_project_type"].astype("string").fillna("other")
    result["exp25_sector_project_type"] = (
        sectors.replace("", "unknown") + "|" + ptype
    ).astype("string")
    result["exp25_state_project_type"] = (
        states.replace("", "unknown") + "|" + ptype
    ).astype("string")

    financial = pd.to_numeric(result.get("financial_progress"), errors="coerce")
    physical = pd.to_numeric(result.get("physical_progress"), errors="coerce")
    result["exp25_financial_progress"] = financial
    result["exp25_financial_physical_gap"] = financial - physical
    return result


def add_milestone_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce")
    result["canonical_project_id"] = result["canonical_project_id"].astype("string")
    status = result.get("milestone_status", pd.Series(None, index=result.index)).astype("string")
    parts = status.str.extract(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
    result["exp25_milestones_achieved"] = pd.to_numeric(parts["done"], errors="coerce")
    result["exp25_milestones_total"] = pd.to_numeric(parts["total"], errors="coerce")
    done = result["exp25_milestones_achieved"]
    total = result["exp25_milestones_total"]
    result["exp25_milestone_ratio"] = (done / total).where(total.gt(0)).clip(0, 1)
    result["exp25_milestones_remaining"] = (
        (total - done).where(total.notna() & done.notna()).clip(lower=0)
    )
    for name in MILESTONE_FEATURES[4:]:
        result[name] = np.nan

    ordered = result.sort_values(["canonical_project_id", "snapshot_date"])
    for _, group in ordered.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group["snapshot_date"]
        achieved = group["exp25_milestones_achieved"]
        ratios = group["exp25_milestone_ratio"]
        months = dates.diff().dt.days / 30.4375
        delta = achieved.diff()
        velocity = ratios.diff().div(months).where(months.gt(0))
        stagnant = pd.Series(
            np.where(delta.notna(), (delta <= 0).astype(float), np.nan),
            index=idx,
        )
        since = pd.Series(np.nan, index=idx, dtype=float)
        previous = np.nan
        last_change = None
        for row_index in idx:
            current = result.at[row_index, "exp25_milestones_achieved"]
            current_date = result.at[row_index, "snapshot_date"]
            if pd.isna(current) or pd.isna(current_date):
                continue
            if pd.isna(previous) or current != previous:
                last_change = current_date
            if last_change is not None:
                since.at[row_index] = max(
                    0.0, (current_date - last_change).days / 30.4375
                )
            previous = current
        result.loc[idx, "exp25_milestone_velocity"] = velocity.to_numpy()
        result.loc[idx, "exp25_milestone_delta"] = delta.to_numpy()
        result.loc[idx, "exp25_milestone_stagnant"] = stagnant.to_numpy()
        result.loc[idx, "exp25_months_since_milestone_change"] = since.to_numpy()
    return result


def enrich_exp25_features(
    frame: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    supervised = add_project_context_features(frame)
    supervised["snapshot_date"] = pd.to_datetime(supervised["snapshot_date"], errors="coerce")
    supervised["canonical_project_id"] = supervised["canonical_project_id"].astype("string")
    if history is None:
        if TRAJECTORIES.exists():
            history = pd.read_csv(
                TRAJECTORIES,
                dtype={"canonical_project_id": "string"},
                low_memory=False,
            )
        else:
            history = frame.copy()
    monthly = add_milestone_features(history)
    lookup = monthly[
        ["canonical_project_id", "snapshot_date", *MILESTONE_FEATURES]
    ].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    supervised = supervised.drop(
        columns=[c for c in MILESTONE_FEATURES if c in supervised],
        errors="ignore",
    )
    return supervised.merge(
        lookup,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _available(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return list(dict.fromkeys(name for name in features if name in frame.columns))


def _selection_split(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    years = sorted(
        int(y)
        for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique()
    )
    if len(years) < 3:
        return None
    for count in (2, 1):
        validation_years = set(years[-count:])
        fitting = train[~train.completion_year.isin(validation_years)].copy()
        validation = train[train.completion_year.isin(validation_years)].copy()
        if (
            fitting.canonical_project_id.nunique() >= 10
            and validation.canonical_project_id.nunique() >= 5
        ):
            return fitting, validation
    return None


def _select_cost_feature_group(
    train: pd.DataFrame,
    base_features: list[str],
    algorithm: str,
) -> tuple[str, list[str], list[dict]]:
    split = _selection_split(train)
    if split is None:
        return "production_contract", list(base_features), [{
            "group": "production_contract",
            "selected": True,
            "reason": "insufficient internal temporal split",
        }]
    fitting, validation = split
    comparisons = []
    for group, additions in FEATURE_GROUPS.items():
        features = _available(train, list(base_features) + list(additions))
        model = _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[algorithm],
            fitting,
            features,
            "actual_cost_overrun_percentage",
        )
        pred = model.predict(validation[features])
        metrics = _regression_metrics(
            validation.actual_cost_overrun_percentage,
            pred,
            validation.sample_weight,
            validation.canonical_project_id,
        )
        comparisons.append({
            "group": group,
            "MAE": metrics["MAE"],
            "features": features,
            "added_features": [name for name in features if name not in base_features],
        })
    winner = min(comparisons, key=lambda item: item["MAE"])
    for item in comparisons:
        item["selected"] = item["group"] == winner["group"]
    return str(winner["group"]), list(winner["features"]), comparisons


def _fit_delay_models(train: pd.DataFrame, features: list[str]) -> dict:
    return _fit_delay_family_models(train, features)


def _delay_predictions(
    models: dict, weights: dict[str, float], frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    return np.maximum(0.0, _blend_predict(models, weights, frame, features))


def _select_delay_feature_group(
    train: pd.DataFrame,
    base_features: list[str],
    production_weights: dict[str, float],
) -> tuple[str, list[str], list[dict]]:
    split = _selection_split(train)
    if split is None:
        return "production_contract", list(base_features), [{
            "group": "production_contract",
            "selected": True,
            "reason": "insufficient internal temporal split",
        }]
    fitting, validation = split
    comparisons = []
    for group, additions in FEATURE_GROUPS.items():
        features = _available(train, list(base_features) + list(additions))
        models = _fit_delay_models(fitting, features)
        pred = _delay_predictions(models, production_weights, validation, features)
        metrics = _regression_metrics(
            validation.actual_delay_days,
            pred,
            validation.sample_weight,
            validation.canonical_project_id,
        )
        comparisons.append({
            "group": group,
            "MAE": metrics["MAE"],
            "features": features,
            "added_features": [name for name in features if name not in base_features],
            "fixed_exp34_weights": dict(production_weights),
        })
    winner = min(comparisons, key=lambda item: item["MAE"])
    for item in comparisons:
        item["selected"] = item["group"] == winner["group"]
    return str(winner["group"]), list(winner["features"]), comparisons


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
    enriched = enrich_exp25_features(
        enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    )
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
    cost_algorithm = selected.get("cost")
    if cost_algorithm not in _regressors(PRODUCTION_COST_SEED):
        raise ValueError(
            f"Exp25 current retest requires a standard production Cost family; got {cost_algorithm!r}."
        )

    production_weights = {
        family: float((metadata.get("delay_blend_weights") or {}).get(family, 0.0))
        for family in FAMILIES
    }
    if abs(sum(production_weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"Exp25 current retest requires normalized Exp34 Delay weights; got {production_weights}."
        )

    base_cost_features = list(contract["cost"])
    base_delay_features = list(contract["delay"])
    cost_group, cost_features, cost_selection = _select_cost_feature_group(
        train, base_cost_features, cost_algorithm
    )
    delay_group, delay_features, delay_selection = _select_delay_feature_group(
        train, base_delay_features, production_weights
    )

    if cost_group == "production_contract":
        cost_model = production_bundle["cost"]
    else:
        cost_model = _fit_pipeline(
            _regressors(PRODUCTION_COST_SEED)[cost_algorithm],
            train,
            cost_features,
            "actual_cost_overrun_percentage",
        )

    delay_models = None
    if delay_group != "production_contract":
        delay_models = _fit_delay_models(train, delay_features)

    compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(compare[base_cost_features])
    prod_delay_pred = np.maximum(
        0.0, production_bundle["delay"].predict(compare[base_delay_features])
    )
    exp_cost_pred = cost_model.predict(compare[cost_features])
    if delay_group == "production_contract":
        exp_delay_pred = prod_delay_pred.copy()
    else:
        exp_delay_pred = _delay_predictions(
            delay_models, production_weights, compare, delay_features
        )

    prod_cost = _regression_metrics(
        compare.actual_cost_overrun_percentage,
        prod_cost_pred,
        compare.sample_weight,
        compare.canonical_project_id,
    )
    exp_cost = _regression_metrics(
        compare.actual_cost_overrun_percentage,
        exp_cost_pred,
        compare.sample_weight,
        compare.canonical_project_id,
    )
    prod_delay = _regression_metrics(
        compare.actual_delay_days,
        prod_delay_pred,
        compare.sample_weight,
        compare.canonical_project_id,
    )
    exp_delay = _regression_metrics(
        compare.actual_delay_days,
        exp_delay_pred,
        compare.sample_weight,
        compare.canonical_project_id,
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

    lookup = {
        _key(row): {feature: row.get(feature) for feature in ALL_ADDED_FEATURES}
        for _, row in test.iterrows()
    }
    train_types = train.exp25_project_type.astype("string")
    test_types = compare.exp25_project_type.astype("string")
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": f"exp25-current-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment",
            "promotion_allowed": False,
            "baseline_contract": "current production = Exp12 Cost + Exp34 Delay",
            "raw_project_name_used_as_feature": False,
            "delay_architecture_policy": "hold Exp34 families and OOF-selected production weights fixed; vary only Exp25 feature group",
            "selected_feature_groups": {
                "cost": cost_group,
                "delay": delay_group,
            },
            "selected_added_features": {
                "cost": [name for name in cost_features if name not in base_cost_features],
                "delay": [name for name in delay_features if name not in base_delay_features],
            },
            "fixed_delay_blend_weights": production_weights,
            "internal_feature_group_comparisons": {
                "cost": cost_selection,
                "delay": delay_selection,
            },
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(compare.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "comparison_cohort": "shared Exp12-comparable production cohort",
            "selected_cost_feature_group": cost_group,
            "selected_delay_feature_group": delay_group,
            "training_milestone_snapshot_share": float(
                train.exp25_milestone_ratio.notna().mean()
            ),
            "test_milestone_snapshot_share": float(
                compare.exp25_milestone_ratio.notna().mean()
            ),
            "training_typed_project_share": float(train_types.ne("other").mean()),
            "test_typed_project_share": float(test_types.ne("other").mean()),
            "training_state_share": float(train.state.notna().mean())
            if "state" in train
            else 0.0,
            "test_state_share": float(compare.state.notna().mean())
            if "state" in compare
            else 0.0,
            "decision": verdict,
        },
        "runtime_state": {
            "cost_model": cost_model,
            "delay_models": delay_models,
            "production_delay_model": production_bundle["delay"],
            "delay_weights": production_weights,
            "cost_features": cost_features,
            "delay_features": delay_features,
            "cost_group": cost_group,
            "delay_group": delay_group,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[
        frame.apply(lambda row: _key(row) in state["comparable"], axis=1)
    ].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError(
            "No Exp25 current-production context representation is available for this snapshot."
        )
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(
        state["cost_model"].predict(one.reindex(columns=state["cost_features"]))[0]
    )
    if state["delay_group"] == "production_contract":
        delay = float(
            state["production_delay_model"].predict(
                one.reindex(columns=state["delay_features"])
            )[0]
        )
    else:
        delay = float(
            _delay_predictions(
                state["delay_models"],
                state["delay_weights"],
                one,
                state["delay_features"],
            )[0]
        )
    return {
        "predicted_cost_overrun": round(cost, 4),
        "predicted_delay_days": round(max(0.0, delay), 4),
        "cost_feature_group": state["cost_group"],
        "delay_feature_group": state["delay_group"],
    }
