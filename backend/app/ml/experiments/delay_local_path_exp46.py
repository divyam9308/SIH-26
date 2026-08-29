"""Experiment 46: causal local Delay path, stagnation and change points.

Only the AFT route's feature representation changes.  Production's Exp32
remaining-time target, three-family weights, Exp33 rolling-OOF residual
calibration method, evidence-only route and exact Exp34 fallback are retained.
"""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.exp35_aft_residual_combo import (
    _aft_remaining_prediction,
    _corrections,
    _delay_aft_calibration_oof,
    _delay_from_remaining,
    _fit_aft_family_models,
    _public_calibration,
    _remaining_frame,
)
from backend.app.ml.experiments.path_oof_delay_exp34 import enrich_path_dependence
from backend.app.ml.experiments.prediction_ledger import (
    assert_prediction_ledger_matches_cohort,
    build_prediction_ledger,
    write_experiment_prediction_ledger,
)
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    _production_cost_evaluation_rows,
    enrich_supervised_for_production,
)
from backend.app.ml.production_exp35_baseline import (
    CALIBRATION_GATE_FEATURE,
    AFTResidualDelayModel,
    _select_aft_calibration_projects,
)

EXPERIMENT_ID = "exp_46"
EXPERIMENT_SEQUENCE = 46
EXPERIMENT_NAME = "Local Delay path, stagnation and change-point features"
EXPERIMENT_SCOPE = "delay"
HYPOTHESIS = (
    "Recent schedule movement, financial-execution slowdown, state stagnation, "
    "reporting irregularity and local regime shifts add remaining-time signal "
    "beyond Exp34 cumulative path summaries."
)

EXP46_FEATURES = [
    "exp46_latest_schedule_revision_days",
    "exp46_absolute_latest_schedule_revision_days",
    "exp46_months_since_schedule_revision",
    "exp46_schedule_revisions_3m",
    "exp46_schedule_revisions_6m",
    "exp46_schedule_inter_revision_cv",
    "exp46_max_schedule_revisions_any_6m",
    "exp46_schedule_same_direction_streak",
    "exp46_schedule_reversal_flag",
    "exp46_slippage_momentum",
    "exp46_financial_momentum",
    "exp46_expenditure_stagnation_streak",
    "exp46_months_since_expenditure_increase",
    "exp46_unchanged_completion_streak",
    "exp46_unchanged_cost_streak",
    "exp46_months_since_meaningful_state_update",
    "exp46_stale_state_fraction_6m",
    "exp46_days_since_previous_report",
    "exp46_recent_median_reporting_gap_days",
    "exp46_recent_max_reporting_gap_days",
    "exp46_recent_skipped_months",
    "exp46_reporting_gap_cv",
    "exp46_schedule_change_point_score",
    "exp46_schedule_change_point_direction",
    "exp46_months_since_schedule_change_point",
    "exp46_schedule_change_point_slope_difference",
    "exp46_financial_change_point_score",
    "exp46_financial_change_point_direction",
    "exp46_months_since_financial_change_point",
]

# Exp12 already computes these exact past-only local derivatives.  They were
# selected for Cost, but are not in the production Delay feature contract.
# Reuse them directly instead of creating synonymous Exp46 columns.
REUSED_EXP12_DELAY_FEATURES = [
    "exp12_slippage_velocity_3m",
    "exp12_slippage_velocity_6m",
    "exp12_slippage_velocity_12m",
    "exp12_slippage_acceleration",
    "exp12_slippage_volatility_6m",
    "exp12_expenditure_ratio_velocity_3m",
    "exp12_expenditure_ratio_velocity_6m",
    "exp12_expenditure_ratio_velocity_12m",
    "exp12_expenditure_ratio_acceleration",
]

SOURCE_COLUMNS = {
    "canonical_project_id",
    "snapshot_date",
    "planned_completion_date",
    "revised_completion_date",
    "schedule_slippage_days",
    "approved_cost_cr",
    "revised_cost_cr",
    "cumulative_expenditure_cr",
}
FORBIDDEN_INPUTS = {
    "completion_date",
    "actual_completion_date",
    "actual_delay_days",
    "physical_progress",
}


def _months(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    return float((later - earlier).days / 30.4375)


def _window_positions(dates: list[pd.Timestamp], current: pd.Timestamp, months: int) -> list[int]:
    days = int(round(months * 30.4375))
    return [index for index, date in enumerate(dates) if 0 <= (current - date).days <= days]


def _slope(dates: list[pd.Timestamp], values: list[float], positions: list[int]) -> float:
    finite = [position for position in positions if np.isfinite(values[position])]
    if len(finite) < 2:
        return 0.0
    first = dates[finite[0]]
    x = np.array([(dates[position] - first).days / 30.4375 for position in finite], dtype=float)
    y = np.array([values[position] for position in finite], dtype=float)
    if np.ptp(x) <= 1e-12:
        return 0.0
    centered = x - x.mean()
    return float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))


def _previous_window_slope(dates: list[pd.Timestamp], values: list[float], current: pd.Timestamp) -> float:
    positions = [index for index, date in enumerate(dates) if 91 < (current - date).days <= 183]
    return _slope(dates, values, positions)


def _change_point(dates: list[pd.Timestamp], values: list[float]) -> tuple[float, float, float, float]:
    if len(values) < 6:
        return 0.0, 0.0, -1.0, 0.0
    current = dates[-1]
    positions = [i for i in _window_positions(dates, current, 24) if np.isfinite(values[i])]
    if len(positions) < 6:
        return 0.0, 0.0, -1.0, 0.0
    local_dates = [dates[i] for i in positions]
    array = np.asarray([values[i] for i in positions], dtype=float)
    median = float(np.median(array))
    scale = max(float(np.median(np.abs(array - median))) * 1.4826, 1.0)
    best = (0.0, 0.0, -1.0, 0.0)
    for split in range(3, len(array) - 2):
        shift = float(np.median(array[split:]) - np.median(array[:split]))
        score = abs(shift) / scale
        if score > best[0] + 1e-12:
            pre = _slope(local_dates, array.tolist(), list(range(split)))
            post = _slope(local_dates, array.tolist(), list(range(split, len(array))))
            best = (score, float(np.sign(shift)), _months(current, local_dates[split]), post - pre)
    return best


def _canonical_history(history: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(SOURCE_COLUMNS.difference(history.columns))
    if missing:
        raise ValueError("Exp46 history is missing: " + ", ".join(missing))
    frame = history.copy()
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    for column in ("planned_completion_date", "revised_completion_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("schedule_slippage_days", "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"])
    # Tie-breaking must itself be prediction-time safe: hash only declared as-of
    # source fields, never labels or completion outcomes carried by the history.
    hash_frame = frame.reindex(columns=sorted(SOURCE_COLUMNS)).astype("string").fillna("<NA>")
    frame["_exp46_tie"] = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(np.uint64)
    frame = frame.sort_values(
        ["canonical_project_id", "snapshot_date", "_exp46_tie"], kind="mergesort"
    ).drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    return frame.drop(columns="_exp46_tie").reset_index(drop=True)


def _streak(changed: list[bool]) -> int:
    count = 0
    for value in reversed(changed):
        if value:
            break
        count += 1
    return count


def engineer_local_delay_history(history: pd.DataFrame) -> pd.DataFrame:
    """Build every local feature from the current/past official-report prefix."""
    frame = _canonical_history(history)
    records: list[dict[str, object]] = []
    for _, group in frame.groupby("canonical_project_id", sort=False):
        dates: list[pd.Timestamp] = []
        slippage: list[float] = []
        financial: list[float] = []
        schedule_event_dates: list[pd.Timestamp] = []
        schedule_moves: list[float] = []
        spend_event_dates: list[pd.Timestamp] = []
        state_event_dates: list[pd.Timestamp] = []
        report_gaps: list[float] = []
        completion_changed: list[bool] = []
        cost_changed: list[bool] = []
        spend_changed: list[bool] = []
        state_stale: list[bool] = []
        previous_effective = pd.NaT
        previous_cost = math.nan
        previous_financial = math.nan

        for _, row in group.iterrows():
            current = pd.Timestamp(row["snapshot_date"])
            dates.append(current)
            if len(dates) > 1:
                report_gaps.append(float((dates[-1] - dates[-2]).days))

            planned = row["planned_completion_date"]
            revised = row["revised_completion_date"]
            effective = revised if pd.notna(revised) else planned
            slip = float(row["schedule_slippage_days"]) if pd.notna(row["schedule_slippage_days"]) else math.nan
            if not np.isfinite(slip) and pd.notna(effective) and pd.notna(planned):
                slip = float((pd.Timestamp(effective) - pd.Timestamp(planned)).days)
            slippage.append(slip)

            revised_cost = float(row["revised_cost_cr"]) if pd.notna(row["revised_cost_cr"]) else math.nan
            approved = float(row["approved_cost_cr"]) if pd.notna(row["approved_cost_cr"]) else math.nan
            expenditure = float(row["cumulative_expenditure_cr"]) if pd.notna(row["cumulative_expenditure_cr"]) else math.nan
            denominator = revised_cost if np.isfinite(revised_cost) and revised_cost > 0 else approved
            ratio = expenditure / denominator if np.isfinite(expenditure) and np.isfinite(denominator) and denominator > 0 else math.nan
            financial.append(ratio * 100.0 if np.isfinite(ratio) else math.nan)

            completion_move = 0.0
            completion_is_changed = False
            if pd.notna(effective) and pd.notna(previous_effective):
                completion_move = float((pd.Timestamp(effective) - pd.Timestamp(previous_effective)).days)
                completion_is_changed = abs(completion_move) > 0
                if completion_is_changed:
                    schedule_event_dates.append(current)
                    schedule_moves.append(completion_move)
            completion_changed.append(completion_is_changed)
            if pd.notna(effective):
                previous_effective = pd.Timestamp(effective)

            cost_is_changed = bool(np.isfinite(revised_cost) and np.isfinite(previous_cost) and abs(revised_cost - previous_cost) > 1e-9)
            cost_changed.append(cost_is_changed)
            if np.isfinite(revised_cost):
                previous_cost = revised_cost

            spend_is_changed = bool(
                np.isfinite(financial[-1]) and np.isfinite(previous_financial) and financial[-1] - previous_financial > 0.01
            )
            spend_changed.append(spend_is_changed)
            if spend_is_changed:
                spend_event_dates.append(current)
            if np.isfinite(financial[-1]):
                previous_financial = financial[-1]
            state_changed = completion_is_changed or cost_is_changed or spend_is_changed
            if state_changed:
                state_event_dates.append(current)
            state_stale.append(not state_changed if len(dates) > 1 else False)

            record: dict[str, object] = {
                "canonical_project_id": row["canonical_project_id"],
                "snapshot_date": current,
                **{feature: 0.0 for feature in EXP46_FEATURES},
                "exp46_months_since_schedule_revision": -1.0,
                "exp46_months_since_expenditure_increase": -1.0,
                "exp46_months_since_meaningful_state_update": -1.0,
                "exp46_months_since_schedule_change_point": -1.0,
                "exp46_months_since_financial_change_point": -1.0,
                "exp46_days_since_previous_report": report_gaps[-1] if report_gaps else -1.0,
                "exp46_unchanged_completion_streak": _streak(completion_changed),
                "exp46_unchanged_cost_streak": _streak(cost_changed),
                "exp46_expenditure_stagnation_streak": _streak(spend_changed),
            }

            if schedule_moves:
                moves = np.asarray(schedule_moves, dtype=float)
                record["exp46_latest_schedule_revision_days"] = moves[-1]
                record["exp46_absolute_latest_schedule_revision_days"] = abs(moves[-1])
                record["exp46_months_since_schedule_revision"] = _months(current, schedule_event_dates[-1])
                record["exp46_schedule_revisions_3m"] = sum((current - date).days <= 92 for date in schedule_event_dates)
                record["exp46_schedule_revisions_6m"] = sum((current - date).days <= 183 for date in schedule_event_dates)
                max6 = max(
                    sum(0 <= (end - date).days <= 183 for date in schedule_event_dates)
                    for end in schedule_event_dates
                )
                record["exp46_max_schedule_revisions_any_6m"] = max6
                streak = 1
                signs = np.sign(moves)
                for position in range(len(signs) - 2, -1, -1):
                    if signs[position] != signs[-1]:
                        break
                    streak += 1
                record["exp46_schedule_same_direction_streak"] = streak
                record["exp46_schedule_reversal_flag"] = float(len(signs) >= 2 and signs[-1] != signs[-2])
                if len(schedule_event_dates) >= 2:
                    intervals = np.diff(np.asarray([date.value for date in schedule_event_dates], dtype=np.int64)) / 86_400_000_000_000 / 30.4375
                    mean_interval = float(np.mean(intervals))
                    record["exp46_schedule_inter_revision_cv"] = float(np.std(intervals) / mean_interval) if mean_interval > 1e-12 else 0.0

            current_slip_slope = _slope(dates, slippage, _window_positions(dates, current, 3))
            current_financial_slope = _slope(dates, financial, _window_positions(dates, current, 3))
            record["exp46_slippage_momentum"] = current_slip_slope - _previous_window_slope(dates, slippage, current)
            record["exp46_financial_momentum"] = current_financial_slope - _previous_window_slope(dates, financial, current)

            if spend_event_dates:
                record["exp46_months_since_expenditure_increase"] = _months(current, spend_event_dates[-1])
            if state_event_dates:
                record["exp46_months_since_meaningful_state_update"] = _months(current, state_event_dates[-1])
            local_stale = [state_stale[i] for i in _window_positions(dates, current, 6)]
            record["exp46_stale_state_fraction_6m"] = float(np.mean(local_stale)) if local_stale else 0.0

            recent_gap_positions = [i for i in range(1, len(dates)) if (current - dates[i]).days <= 366]
            recent_gaps = [float((dates[i] - dates[i - 1]).days) for i in recent_gap_positions]
            if recent_gaps:
                record["exp46_recent_median_reporting_gap_days"] = float(np.median(recent_gaps))
                record["exp46_recent_max_reporting_gap_days"] = float(np.max(recent_gaps))
                record["exp46_recent_skipped_months"] = float(sum(max(0, round(gap / 30.4375) - 1) for gap in recent_gaps))
                mean_gap = float(np.mean(recent_gaps))
                record["exp46_reporting_gap_cv"] = float(np.std(recent_gaps) / mean_gap) if mean_gap > 1e-12 else 0.0

            schedule_cp = _change_point(dates, slippage)
            financial_cp = _change_point(dates, financial)
            record["exp46_schedule_change_point_score"] = schedule_cp[0]
            record["exp46_schedule_change_point_direction"] = schedule_cp[1]
            record["exp46_months_since_schedule_change_point"] = schedule_cp[2]
            record["exp46_schedule_change_point_slope_difference"] = schedule_cp[3]
            record["exp46_financial_change_point_score"] = financial_cp[0]
            record["exp46_financial_change_point_direction"] = financial_cp[1]
            record["exp46_months_since_financial_change_point"] = financial_cp[2]
            records.append(record)

    return pd.DataFrame.from_records(records, columns=["canonical_project_id", "snapshot_date", *EXP46_FEATURES])


def enrich_local_delay_path(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        history = pd.read_csv(TRAJECTORIES, low_memory=False) if TRAJECTORIES.exists() else supervised.copy()
    engineered = engineer_local_delay_history(history)
    rows = supervised.copy()
    rows["canonical_project_id"] = rows["canonical_project_id"].astype("string")
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    result = rows.merge(engineered, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")
    if len(result) != len(rows):
        raise AssertionError("Exp46 feature engineering changed the supervised cohort")
    for feature in EXP46_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(
            -1.0 if feature.startswith("exp46_months_since_") or feature == "exp46_days_since_previous_report" else 0.0
        )
    return result


def _metric(rows: pd.DataFrame, prediction: np.ndarray) -> dict:
    return _regression_metrics(rows.actual_delay_days, prediction, rows.sample_weight, rows.canonical_project_id)


def _route_metrics(rows: pd.DataFrame, production: np.ndarray, candidate: np.ndarray) -> dict:
    if rows.empty:
        return {"projects": 0, "snapshots": 0, "production_delay_mae": None, "experiment_delay_mae": None}
    weighted = assign_project_balanced_weights(rows)
    positions = weighted["_full_position"].to_numpy(int)
    return {
        "projects": int(weighted.canonical_project_id.nunique()),
        "snapshots": int(len(weighted)),
        "production_delay_mae": _metric(weighted, production[positions])["MAE"],
        "experiment_delay_mae": _metric(weighted, candidate[positions])["MAE"],
    }


def _diagnostics(rows: pd.DataFrame) -> dict:
    work = rows.copy()
    work["_prod_error"] = (work["production_delay_prediction"] - work["actual_delay_days"]).abs()
    work["_exp_error"] = (work["experiment_delay_prediction"] - work["actual_delay_days"]).abs()
    project = work.groupby("canonical_project_id").agg(prod=("_prod_error", "mean"), exp=("_exp_error", "mean"))
    result = {
        "median_per_project_mae": {
            "production": float(project["prod"].median()),
            "experiment": float(project["exp"].median()),
        },
        "p90_per_project_mae": {
            "production": float(project["prod"].quantile(.9)),
            "experiment": float(project["exp"].quantile(.9)),
        },
        "absolute_error_p90": {
            "production": float(work["_prod_error"].quantile(.9)),
            "experiment": float(work["_exp_error"].quantile(.9)),
        },
        "lifecycle_mae": {},
    }
    for stage in ("early", "mid", "late", "very_late"):
        part = work[work.lifecycle_stage.eq(stage)] if "lifecycle_stage" in work else work.iloc[0:0]
        if part.empty:
            result["lifecycle_mae"][stage] = {"projects": 0, "snapshots": 0, "production_mae": None, "experiment_mae": None}
        else:
            weights = part.sample_weight.to_numpy(float)
            result["lifecycle_mae"][stage] = {
                "projects": int(part.canonical_project_id.nunique()), "snapshots": int(len(part)),
                "production_mae": float(np.average(part["_prod_error"], weights=weights)),
                "experiment_mae": float(np.average(part["_exp_error"], weights=weights)),
            }
    return result


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_local_delay_path(enrich_path_dependence(enrich_supervised_for_production(data.copy())))
    enriched["completion_year"] = pd.to_numeric(enriched["completion_year"], errors="coerce")
    enriched["snapshot_date"] = pd.to_datetime(enriched["snapshot_date"], errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    if set(train.canonical_project_id) & set(test.canonical_project_id):
        raise ValueError("Exp46 train/holdout project overlap")

    production_delay_model = production_bundle["delay"]
    if not isinstance(production_delay_model, AFTResidualDelayModel):
        raise TypeError("Exp46 requires current production AFTResidualDelayModel")
    base_features = list(production_delay_model.features)
    duplicated = sorted(set(base_features) & set(EXP46_FEATURES))
    if duplicated:
        raise ValueError("Exp46 duplicates production features: " + ", ".join(duplicated))
    unavailable_reused = [feature for feature in REUSED_EXP12_DELAY_FEATURES if feature not in train]
    if unavailable_reused:
        raise ValueError("Exp46 expected existing Exp12 causal features: " + ", ".join(unavailable_reused))
    reused = [feature for feature in REUSED_EXP12_DELAY_FEATURES if feature not in base_features]
    features = list(dict.fromkeys(base_features + reused + EXP46_FEATURES))
    weights = dict(production_delay_model.weights)
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Exp46 production AFT weights are not normalized: {weights}")

    train_delay = _remaining_frame(train)
    calibration, oof = _delay_aft_calibration_oof(train_delay, features, weights)
    aft_models = _fit_aft_family_models(train_delay, features)

    compare = _production_cost_evaluation_rows(test).copy()
    gate_projects = _select_aft_calibration_projects(compare)
    compare[CALIBRATION_GATE_FEATURE] = compare.canonical_project_id.astype("string").isin(gate_projects)
    compare = assign_project_balanced_weights(compare).reset_index(drop=True)
    production_delay = np.maximum(0.0, production_delay_model.predict(compare))
    candidate_delay = production_delay.copy()
    route_mask = production_delay_model._aft_eligible(compare).to_numpy(bool)
    if route_mask.any():
        positions = np.flatnonzero(route_mask)
        route = compare.iloc[positions].copy()
        remaining = _aft_remaining_prediction(aft_models, weights, route, features)
        raw_delay = _delay_from_remaining(route, remaining)
        candidate_delay[positions] = np.maximum(0.0, raw_delay + _corrections(route, raw_delay, calibration))
    if not np.array_equal(candidate_delay[~route_mask], production_delay[~route_mask]):
        raise AssertionError("Exp46 changed the Exp34 fallback")

    production_cost = production_bundle["cost"].predict(compare)
    candidate_cost = production_cost.copy()
    if not np.array_equal(production_cost, candidate_cost):
        raise AssertionError("Exp46 changed Cost predictions")

    prod_metrics = _metric(compare, production_delay)
    exp_metrics = _metric(compare, candidate_delay)
    improvement = float(prod_metrics["MAE"]) - float(exp_metrics["MAE"])
    improvement_pct = improvement / float(prod_metrics["MAE"]) * 100.0
    scientific_verdict = "PROMOTION CANDIDATE" if improvement_pct > 0 else "DO NOT PROMOTE"

    scored = compare.copy()
    scored["production_delay_prediction"] = production_delay
    scored["experiment_delay_prediction"] = candidate_delay
    scored["experiment_route"] = np.where(route_mask, "exp46_aft", "exp34_fallback")
    scored["_full_position"] = np.arange(len(scored))
    statistics = paired_project_mae_comparison(
        scored,
        actual="actual_delay_days",
        baseline_prediction="production_delay_prediction",
        candidate_prediction="experiment_delay_prediction",
        bootstrap_samples=5000,
        seed=46000 + int(training_end),
    )
    aft_metrics = _route_metrics(scored[route_mask].copy(), production_delay, candidate_delay)
    fallback_metrics = _route_metrics(scored[~route_mask].copy(), production_delay, candidate_delay)
    fallback_prod = fallback_metrics["production_delay_mae"]
    fallback_exp = fallback_metrics["experiment_delay_mae"]
    if fallback_prod is not None and not math.isclose(
        float(fallback_prod), float(fallback_exp), rel_tol=0.0, abs_tol=1e-9
    ):
        raise AssertionError("Exp46 fallback metrics are not identical")

    # Production's published 688/33 project split is project-level: 688
    # projects are admitted to the fixed AFT evidence gate and the remaining
    # 33 are fallback-only.  Some gated projects still use fallback on
    # individual snapshots where planned-completion evidence is unavailable,
    # so counting unique IDs among all fallback *rows* is intentionally larger
    # and belongs in fallback_route["projects"], not fallback_projects.
    aft_project_ids = set(scored.loc[route_mask, "canonical_project_id"].astype("string"))
    all_project_ids = set(scored["canonical_project_id"].astype("string"))
    fallback_only_projects = len(all_project_ids - aft_project_ids)

    run_id = f"exp46-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    window = f"{training_start}_{training_end}"
    ledger = build_prediction_ledger(
        scored,
        experiment_id=EXPERIMENT_ID,
        window=window,
        production_delay_prediction=production_delay,
        experiment_delay_prediction=candidate_delay,
        extra_columns=[
            "completion_year", "lifecycle_stage", "sector", "implementing_agency", "state",
            "project_size_category", "approved_cost_cr", "cost_escalation_percentage",
            "schedule_slippage_days", "duration_ratio", "revised_cost_cr",
            "cumulative_expenditure_cr", "exp12_history_12m", "exp34_observations_seen",
            "parser_family", "experiment_route",
        ],
    )
    assert_prediction_ledger_matches_cohort(ledger, compare)
    persisted = write_experiment_prediction_ledger(
        ledger,
        experiment_id=EXPERIMENT_ID,
        window=window,
        run_id=run_id,
        extra_manifest={
            "primary_target": "delay",
            "execution_verdict": "EXECUTION VALID",
            "scientific_verdict": scientific_verdict,
            "changed_dimension": "feature_set",
            "bootstrap_samples": 5000,
            "cost_unchanged": True,
            "fallback_unchanged": True,
        },
    )

    lookup = {
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): float(prediction)
        for (_, row), prediction in zip(scored.iterrows(), candidate_delay)
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE, "run_id": run_id, "model_role": "experiment",
            "promotion_allowed": False, "changed_dimension": "feature_set", "hypothesis": HYPOTHESIS,
            "new_features": EXP46_FEATURES,
            "reused_existing_causal_features": reused,
            "incremental_feature_count": len(reused) + len(EXP46_FEATURES),
            "fixed_production_aft_weights": weights,
            "calibration_method": "production Exp33 rolling-OOF weighted-median residual calibration, refit for challenger",
            "rolling_oof": oof, "calibration": _public_calibration(calibration),
            "future_holdout_used_for_selection_routing_or_calibration": False,
            "execution_verdict": "EXECUTION VALID", "scientific_verdict": scientific_verdict,
            "ledger_path": str(persisted["ledger_path"]),
            "ledger_manifest_path": str(persisted["manifest_path"]),
            "cohort_fingerprint": persisted["manifest"]["cohort_fingerprint"],
            "ledger_fingerprint": persisted["manifest"]["ledger_fingerprint"],
        },
        "overall_comparison": {
            "production_delay_mae": prod_metrics["MAE"], "experiment_delay_mae": exp_metrics["MAE"],
            "absolute_delay_mae_improvement": round(improvement, 6),
            "delay_improvement_percentage": round(improvement_pct, 6),
            "production_cost_mae": _regression_metrics(compare.actual_cost_overrun_percentage, production_cost, compare.sample_weight, compare.canonical_project_id)["MAE"],
            "experiment_cost_mae": _regression_metrics(compare.actual_cost_overrun_percentage, candidate_cost, compare.sample_weight, compare.canonical_project_id)["MAE"],
            "cost_predictions_identical": True,
            "comparison_test_projects": int(compare.canonical_project_id.nunique()),
            "comparison_test_snapshots": int(len(compare)),
            "aft_projects": len(aft_project_ids),
            "fallback_projects": fallback_only_projects,
            "aft_snapshots": int(route_mask.sum()), "fallback_snapshots": int((~route_mask).sum()),
            "aft_route": aft_metrics, "fallback_route": fallback_metrics,
            "paired_project_bootstrap": statistics, "diagnostics": _diagnostics(scored),
            "execution_verdict": "EXECUTION VALID", "scientific_verdict": scientific_verdict,
        },
        "state": {"lookup": lookup, "features": features, "aft_models": aft_models, "calibration": calibration},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    keys = set(state.get("lookup", {}))
    mask = [
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) in keys
        for _, row in frame.iterrows()
    ]
    return assign_project_balanced_weights(frame.loc[mask].copy())


def predict_project(row: pd.Series, state: dict) -> dict:
    key = (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
    if key not in state.get("lookup", {}):
        raise ValueError("Exp46 row is outside the frozen comparison cohort")
    return {"delay_days": float(state["lookup"][key])}
