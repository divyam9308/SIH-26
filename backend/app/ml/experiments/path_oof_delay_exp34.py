"""Experiment 34: Exp29 path dependence + Exp31 rolling-OOF ensemble, Delay only.

Scientific contract
-------------------
* Cost is retained from the exact current production model (promoted Exp12).
* Delay is the only challenged target.
* Exp29 dense full-lifecycle path features are constructed causally from the
  official monthly trajectory table using only current/prior project history.
* Exp31's non-negative ExtraTrees/LightGBM/XGBoost blend weights are selected
  exclusively from rolling out-of-fold years inside the training window.
* The 2022-2025 future holdout is never used for feature or blend selection.
"""
from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from backend.app.ml.monthly_lifecycle import TRAJECTORIES
from backend.app.ml.monthly_training import (
    _fit_pipeline,
    _regression_metrics,
    _regressors,
    temporal_project_split,
)
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
    target_feature_contract,
)

EXPERIMENT_ID = "exp_34"
EXPERIMENT_NAME = "Exp29 path dependence + Exp31 OOF ensemble (Delay only)"
EXPERIMENT_SCOPE = "delay"
EXPERIMENT_SEQUENCE = 34
DELAY_SEED = 26204
MAX_FOLDS = 3
GRID_STEP = 0.1
FAMILIES = ("extra_trees", "lightgbm", "xgboost")

PATH_FEATURES = [
    "exp34_observations_seen",
    "exp34_months_observed",
    "exp34_cost_revision_count",
    "exp34_schedule_revision_count",
    "exp34_cumulative_abs_cost_revision_pct",
    "exp34_max_cost_escalation",
    "exp34_cost_recovery_from_peak",
    "exp34_max_schedule_slippage",
    "exp34_delay_recovery_from_peak",
    "exp34_slippage_positive_share",
    "exp34_cost_overrun_positive_share",
    "exp34_cost_worsening_share",
    "exp34_delay_worsening_share",
    "exp34_months_since_first_cost_revision",
    "exp34_months_since_first_schedule_revision",
]


def _gain(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0 if base else 0.0


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _expanding_mean(values: pd.Series, groups: pd.Series) -> pd.Series:
    return (
        values.groupby(groups, sort=False)
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _path_history(history: pd.DataFrame) -> pd.DataFrame:
    """Build the Exp29-style cumulative path representation causally."""
    hist = history.copy()
    hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"], errors="coerce")
    hist["canonical_project_id"] = hist["canonical_project_id"].astype("string")
    hist = hist.dropna(subset=["canonical_project_id", "snapshot_date"]).sort_values(
        ["canonical_project_id", "snapshot_date"]
    )
    gid = hist["canonical_project_id"]
    for col in ("revised_cost_cr", "cost_escalation_percentage", "schedule_slippage_days"):
        if col not in hist:
            hist[col] = np.nan
        hist[col] = pd.to_numeric(hist[col], errors="coerce")

    hist["exp34_observations_seen"] = hist.groupby("canonical_project_id").cumcount() + 1
    first_snapshot = hist.groupby("canonical_project_id")["snapshot_date"].transform("min")
    hist["exp34_months_observed"] = (hist["snapshot_date"] - first_snapshot).dt.days / 30.4375

    prev_cost = hist.groupby("canonical_project_id")["revised_cost_cr"].shift(1)
    cost_change = (
        hist["revised_cost_cr"].notna()
        & prev_cost.notna()
        & hist["revised_cost_cr"].sub(prev_cost).abs().gt(1e-9)
    )
    prev_slip = hist.groupby("canonical_project_id")["schedule_slippage_days"].shift(1)
    schedule_change = (
        hist["schedule_slippage_days"].notna()
        & prev_slip.notna()
        & hist["schedule_slippage_days"].sub(prev_slip).abs().gt(1e-9)
    )
    hist["exp34_cost_revision_count"] = cost_change.astype(int).groupby(gid).cumsum()
    hist["exp34_schedule_revision_count"] = schedule_change.astype(int).groupby(gid).cumsum()

    pct_revision = (
        hist["revised_cost_cr"]
        .sub(prev_cost)
        .abs()
        .div(prev_cost.abs().replace(0, np.nan))
        .mul(100)
        .fillna(0)
    )
    hist["exp34_cumulative_abs_cost_revision_pct"] = pct_revision.groupby(gid).cumsum()
    hist["exp34_max_cost_escalation"] = hist.groupby("canonical_project_id")["cost_escalation_percentage"].cummax()
    hist["exp34_cost_recovery_from_peak"] = hist["exp34_max_cost_escalation"] - hist["cost_escalation_percentage"]
    hist["exp34_max_schedule_slippage"] = hist.groupby("canonical_project_id")["schedule_slippage_days"].cummax()
    hist["exp34_delay_recovery_from_peak"] = hist["exp34_max_schedule_slippage"] - hist["schedule_slippage_days"]

    slip_positive = hist["schedule_slippage_days"].gt(0).astype(float).where(hist["schedule_slippage_days"].notna())
    cost_positive = hist["cost_escalation_percentage"].gt(0).astype(float).where(hist["cost_escalation_percentage"].notna())
    group_change = gid.ne(gid.shift())
    cost_worsening = hist["cost_escalation_percentage"].diff().gt(0).astype(float).mask(group_change)
    delay_worsening = hist["schedule_slippage_days"].diff().gt(0).astype(float).mask(group_change)
    hist["exp34_slippage_positive_share"] = _expanding_mean(slip_positive, gid)
    hist["exp34_cost_overrun_positive_share"] = _expanding_mean(cost_positive, gid)
    hist["exp34_cost_worsening_share"] = _expanding_mean(cost_worsening, gid)
    hist["exp34_delay_worsening_share"] = _expanding_mean(delay_worsening, gid)

    first_cost_revision = hist["snapshot_date"].where(cost_change).groupby(gid).transform("min")
    first_schedule_revision = hist["snapshot_date"].where(schedule_change).groupby(gid).transform("min")
    hist["exp34_months_since_first_cost_revision"] = (
        (hist["snapshot_date"] - first_cost_revision).dt.days / 30.4375
    ).fillna(-1)
    hist["exp34_months_since_first_schedule_revision"] = (
        (hist["snapshot_date"] - first_schedule_revision).dt.days / 30.4375
    ).fillna(-1)

    return hist[["canonical_project_id", "snapshot_date", *PATH_FEATURES]].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )


def enrich_path_dependence(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = pd.read_csv(TRAJECTORIES, low_memory=False) if TRAJECTORIES.exists() else supervised.copy()
    engineered = _path_history(history)
    result = supervised.copy()
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce")
    result["canonical_project_id"] = result["canonical_project_id"].astype("string")
    return result.merge(
        engineered,
        on=["canonical_project_id", "snapshot_date"],
        how="left",
        validate="many_to_one",
    )


def _rolling_folds(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    years = sorted(int(y) for y in pd.to_numeric(train["completion_year"], errors="coerce").dropna().unique())
    folds: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    completion_year = pd.to_numeric(train["completion_year"], errors="coerce")
    for year in reversed(years[1:]):
        fitting = train[completion_year.lt(year)].copy()
        validation = train[completion_year.eq(year)].copy()
        if fitting["canonical_project_id"].nunique() >= 10 and validation["canonical_project_id"].nunique() >= 3:
            folds.append((fitting, validation, year))
        if len(folds) >= MAX_FOLDS:
            break
    return list(reversed(folds))


def _weight_grid(step: float = GRID_STEP) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    rows: list[dict[str, float]] = []
    for a in range(units + 1):
        for b in range(units + 1 - a):
            c = units - a - b
            rows.append({FAMILIES[0]: a / units, FAMILIES[1]: b / units, FAMILIES[2]: c / units})
    return rows


def _oof_delay_weights(train: pd.DataFrame, features: list[str]) -> tuple[dict[str, float], dict]:
    folds = _rolling_folds(train)
    if len(folds) < 2:
        raise ValueError("Experiment 34 requires at least two historical rolling OOF folds.")
    chunks: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for fitting, validation, year in folds:
        chunk = validation[["actual_delay_days", "sample_weight", "canonical_project_id"]].copy()
        diag = {"year": year, "projects": int(validation["canonical_project_id"].nunique())}
        for family in FAMILIES:
            model = _fit_pipeline(_regressors(DELAY_SEED)[family], fitting, features, "actual_delay_days")
            pred = np.maximum(0, model.predict(validation[features]))
            chunk[family] = pred
            diag[family] = _regression_metrics(
                validation["actual_delay_days"], pred, validation["sample_weight"], validation["canonical_project_id"]
            )["MAE"]
        chunks.append(chunk)
        diagnostics.append(diag)

    oof = pd.concat(chunks, ignore_index=True)
    actual = pd.to_numeric(oof["actual_delay_days"], errors="coerce").to_numpy(float)
    sample_weight = pd.to_numeric(oof["sample_weight"], errors="coerce").to_numpy(float)
    best: dict | None = None
    comparisons: list[dict] = []
    for blend in _weight_grid():
        pred = sum(float(blend[name]) * oof[name].to_numpy(float) for name in FAMILIES)
        mae = float(mean_absolute_error(actual, pred, sample_weight=sample_weight))
        row = {"weights": blend, "MAE": mae}
        comparisons.append(row)
        if best is None or mae < best["MAE"]:
            best = row
    assert best is not None
    return dict(best["weights"]), {
        "folds": diagnostics,
        "oof_rows": int(len(oof)),
        "grid_size": len(comparisons),
        "best_oof_mae": float(best["MAE"]),
    }


def _fit_delay_family_models(train: pd.DataFrame, features: list[str]) -> dict:
    return {
        family: _fit_pipeline(_regressors(DELAY_SEED)[family], train, features, "actual_delay_days")
        for family in FAMILIES
    }


def _blend_predict(models: dict, weights: dict[str, float], frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=float)
    for family in FAMILIES:
        pred += float(weights[family]) * models[family].predict(frame[features])
    return pred


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    metadata = dict(production_bundle.get("metadata") or {})
    contract = target_feature_contract(metadata)
    cost_features = list(contract["cost"])
    delay_features = list(dict.fromkeys(list(contract["delay"]) + PATH_FEATURES))

    # Exp31 logic, but applied to the Exp29-enriched Delay representation.
    delay_weights, delay_oof = _oof_delay_weights(train, delay_features)
    delay_models = _fit_delay_family_models(train, delay_features)

    # Cost is intentionally untouched and evaluated on the official Exp12 cohort.
    cost_compare = _production_cost_evaluation_rows(test)
    prod_cost_pred = production_bundle["cost"].predict(cost_compare[cost_features])
    exp_cost_pred = prod_cost_pred.copy()
    prod_cost = _regression_metrics(
        cost_compare["actual_cost_overrun_percentage"], prod_cost_pred,
        cost_compare["sample_weight"], cost_compare["canonical_project_id"]
    )
    exp_cost = _regression_metrics(
        cost_compare["actual_cost_overrun_percentage"], exp_cost_pred,
        cost_compare["sample_weight"], cost_compare["canonical_project_id"]
    )

    prod_delay_pred = np.maximum(0, production_bundle["delay"].predict(test[list(contract["delay"])]))
    exp_delay_pred = np.maximum(0, _blend_predict(delay_models, delay_weights, test, delay_features))
    prod_delay = _regression_metrics(
        test["actual_delay_days"], prod_delay_pred, test["sample_weight"], test["canonical_project_id"]
    )
    exp_delay = _regression_metrics(
        test["actual_delay_days"], exp_delay_pred, test["sample_weight"], test["canonical_project_id"]
    )

    cost_gain = _gain(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _gain(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    if abs(cost_gain) > 1e-12:
        raise AssertionError(f"Delay-only Exp34 changed Cost MAE unexpectedly: {cost_gain}")
    verdict = "PROMOTION CANDIDATE" if delay_gain > 0 else "REGRESSION / DO NOT PROMOTE"

    lookup_features = list(dict.fromkeys(cost_features + delay_features))
    lookup = {
        _key(row): {name: row.get(name) for name in lookup_features}
        for _, row in test.iterrows()
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": f"exp34-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}",
            "model_role": "experiment",
            "promotion_allowed": False,
            "changed_dimension": "delay_path_features_plus_oof_ensemble",
            "component_evidence": ["exp_29", "exp_31"],
            "delay_path_features": PATH_FEATURES,
            "delay_families": list(FAMILIES),
            "selected_delay_weights": delay_weights,
            "rolling_oof": delay_oof,
            "cost_policy": "production_exp12_retained_exactly",
            "future_holdout_used_for_selection": False,
            "decision": verdict,
        },
        "overall_comparison": {
            "production_cost_mae": prod_cost["MAE"],
            "experiment_cost_mae": exp_cost["MAE"],
            "cost_improvement_percentage": round(cost_gain, 4),
            "production_delay_mae": prod_delay["MAE"],
            "experiment_delay_mae": exp_delay["MAE"],
            "delay_improvement_percentage": round(delay_gain, 4),
            "comparison_test_projects": int(test["canonical_project_id"].nunique()),
            "comparison_test_snapshots": int(len(test)),
            "cost_comparison_projects": int(cost_compare["canonical_project_id"].nunique()),
            "path_feature_nonmissing_share": float(test[PATH_FEATURES].notna().any(axis=1).mean()),
            "delay_blend_weights": delay_weights,
            "decision": verdict,
        },
        "runtime_state": {
            "production_cost_model": production_bundle["cost"],
            "cost_features": cost_features,
            "delay_models": delay_models,
            "delay_weights": delay_weights,
            "delay_features": delay_features,
            "lookup": lookup,
            "comparable": set(lookup),
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 34 path history is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    one = candidate.to_frame().T
    cost = float(state["production_cost_model"].predict(one.reindex(columns=state["cost_features"]))[0])
    delay = max(0.0, float(_blend_predict(
        state["delay_models"], state["delay_weights"], one, state["delay_features"]
    )[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4)}
