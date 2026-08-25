from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset, training_as_of_invariants
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors

FEATURES = [
    "approved_cost_cr", "sector_average_delay", "sector_average_cost_overrun", "sector",
    "project_size_category", "cumulative_expenditure_cr", "expenditure_ratio",
    "schedule_slippage_days", "schedule_slippage_ratio", "elapsed_duration_days",
    "planned_duration_days", "duration_ratio", "expected_progress_percentage",
    "revised_cost_cr", "cost_escalation_percentage", "implementing_agency",
    "cost_growth_velocity_3m", "cost_growth_velocity_6m", "cost_acceleration",
    "sector_delay_rate", "sector_cost_overrun_rate", "agency_average_delay",
    "agency_average_cost_overrun", "agency_delay_rate", "agency_cost_overrun_rate",
]

CANDIDATES = {
    "A_no_decay": None,
    "B_15y": 15.0,
    "C_10y": 10.0,
    "D_7y": 7.0,
    "E_5y": 5.0,
    "F_3y": 3.0,
}
VALIDATION_YEARS = [2010, 2012, 2015, 2018, 2019, 2021]
SEED = 26203
TARGET = "actual_cost_overrun_percentage"
OUT = Path("reports/audits/recency_decay_25f_audit.json")


def _slice(data: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return data[data.completion_year.between(start, end)].copy()


def _apply_decay(train: pd.DataFrame, training_end: int, half_life: float | None) -> tuple[pd.DataFrame, dict]:
    out = train.copy()
    base = pd.to_numeric(out.sample_weight, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if half_life is None:
        decay = np.ones(len(out), dtype=float)
    else:
        age = (float(training_end) - pd.to_numeric(out.completion_year, errors="coerce").to_numpy(dtype=float)).clip(min=0.0)
        decay = np.power(0.5, age / float(half_life))
    weighted = base * decay
    if weighted.sum() <= 0:
        raise ValueError("Recency weighting produced zero total training weight")
    weighted *= base.sum() / weighted.sum()
    out["sample_weight"] = weighted
    return out, {
        "half_life_years": half_life,
        "decay_min": round(float(decay.min()), 6),
        "decay_median": round(float(np.median(decay)), 6),
        "decay_max": round(float(decay.max()), 6),
        "normalised_weight_sum": round(float(weighted.sum()), 6),
    }


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, training_end: int, half_life: float | None) -> tuple[dict, pd.DataFrame, dict]:
    weighted_train, decay_info = _apply_decay(train, training_end, half_life)
    model = _fit_pipeline(_regressors(SEED)["extra_trees"], weighted_train, FEATURES, TARGET)
    pred = model.predict(test[FEATURES])
    metrics = _regression_metrics(test[TARGET], pred, test.sample_weight, test.canonical_project_id)
    rows = test[["canonical_project_id", "lifecycle_stage", TARGET, "sample_weight"]].copy()
    rows["predicted_cost_overrun"] = pred
    rows["abs_error"] = (rows.predicted_cost_overrun - rows[TARGET]).abs()
    return metrics, rows, decay_info


def _stage_mae(rows: pd.DataFrame) -> dict:
    result = {}
    for stage in ["early", "mid", "late", "very_late"]:
        part = rows[rows.lifecycle_stage.eq(stage)]
        if part.empty:
            result[stage] = None
            continue
        result[stage] = round(float(np.average(part.abs_error, weights=part.sample_weight)), 3)
    return result


def _project_abs_errors(rows: pd.DataFrame) -> pd.Series:
    def one(group: pd.DataFrame) -> float:
        w = group.sample_weight.to_numpy(dtype=float)
        e = group.abs_error.to_numpy(dtype=float)
        return float(np.average(e, weights=w)) if w.sum() > 0 else float(np.mean(e))
    return rows.groupby("canonical_project_id", sort=False).apply(one)


def _bootstrap_difference(a_rows: pd.DataFrame, b_rows: pd.DataFrame, samples: int = 1000) -> dict:
    a = _project_abs_errors(a_rows)
    b = _project_abs_errors(b_rows)
    common = a.index.intersection(b.index)
    delta = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    rng = np.random.default_rng(26103)
    boot = np.empty(samples, dtype=float)
    for i in range(samples):
        idx = rng.integers(0, len(delta), len(delta))
        boot[i] = float(delta[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "metric": "project_mean_abs_error_no_decay_minus_winner",
        "projects": int(len(common)),
        "point_difference_pp": round(float(delta.mean()), 3),
        "ci95_lower_pp": round(float(lo), 3),
        "ci95_upper_pp": round(float(hi), 3),
        "samples": samples,
    }


def main() -> None:
    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing required 25-feature production columns: {missing}")

    fold_results: dict[str, list[dict]] = {name: [] for name in CANDIDATES}
    for validation_year in VALIDATION_YEARS:
        train = _slice(data, 2001, validation_year - 1)
        validation = _slice(data, validation_year, validation_year)
        if train.canonical_project_id.nunique() < 10 or validation.canonical_project_id.nunique() < 2:
            raise ValueError(f"Insufficient projects for validation year {validation_year}")
        if not training_as_of_invariants(train)["passed"] or not training_as_of_invariants(validation)["passed"]:
            raise ValueError(f"As-of invariant failed for validation year {validation_year}")
        for name, half_life in CANDIDATES.items():
            metrics, rows, decay_info = _fit_predict(train, validation, validation_year - 1, half_life)
            fold_results[name].append({
                "validation_year": validation_year,
                "training_period": [2001, validation_year - 1],
                "training_projects": int(train.canonical_project_id.nunique()),
                "validation_projects": int(validation.canonical_project_id.nunique()),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "stage_mae": _stage_mae(rows),
                "decay": decay_info,
            })

    summaries = {}
    for name, folds in fold_results.items():
        maes = np.array([float(f["MAE"]) for f in folds], dtype=float)
        projects = np.array([int(f["validation_projects"]) for f in folds], dtype=float)
        summaries[name] = {
            "half_life_years": CANDIDATES[name],
            "mean_mae": round(float(maes.mean()), 3),
            "median_mae": round(float(np.median(maes)), 3),
            "worst_fold_mae": round(float(maes.max()), 3),
            "project_count_weighted_mean_mae": round(float(np.average(maes, weights=projects)), 3),
        }

    winner = min(
        summaries,
        key=lambda name: (
            summaries[name]["mean_mae"],
            summaries[name]["median_mae"],
            summaries[name]["worst_fold_mae"],
        ),
    )

    final_train = _slice(data, 2001, 2021)
    final_test = _slice(data, 2022, 2025)
    no_decay_metrics, no_decay_rows, _ = _fit_predict(final_train, final_test, 2021, None)
    winner_metrics, winner_rows, winner_decay = _fit_predict(final_train, final_test, 2021, CANDIDATES[winner])

    recent_train = _slice(data, 2018, 2021)
    recent_metrics, recent_rows, _ = _fit_predict(recent_train, final_test, 2021, None)

    no_decay_mae = float(no_decay_metrics["MAE"])
    winner_mae = float(winner_metrics["MAE"])
    recent_mae = float(recent_metrics["MAE"])
    improvement_pct = (no_decay_mae - winner_mae) / no_decay_mae * 100.0

    report = {
        "audit_type": "temporary_read_only_recency_decay_25_feature",
        "policy": {
            "repository_main_modified": False,
            "feature_count": len(FEATURES),
            "features": FEATURES,
            "algorithm_fixed": "extra_trees",
            "algorithm_seed": SEED,
            "selection_uses_2022_2025": False,
            "selection_metric": "equal-fold mean project-balanced MAE; median then worst-fold tie-break",
            "weight_formula": "existing_project_balanced_sample_weight * 2^(-age_years/half_life_years), renormalised to original total training weight",
            "validation_years": VALIDATION_YEARS,
        },
        "dataset": {
            "rows": int(len(data)),
            "projects": int(data.canonical_project_id.nunique()),
            "completion_year_min": int(data.completion_year.min()),
            "completion_year_max": int(data.completion_year.max()),
        },
        "candidates": summaries,
        "folds": fold_results,
        "selected_candidate": winner,
        "selected_half_life_years": CANDIDATES[winner],
        "final_untouched_holdout": {
            "training_period": [2001, 2021],
            "testing_period": [2022, 2025],
            "training_projects": int(final_train.canonical_project_id.nunique()),
            "test_projects": int(final_test.canonical_project_id.nunique()),
            "no_decay": {"metrics": no_decay_metrics, "stage_mae": _stage_mae(no_decay_rows)},
            "selected_decay": {"metrics": winner_metrics, "stage_mae": _stage_mae(winner_rows), "decay": winner_decay},
            "recent_only_2018_2021": {"metrics": recent_metrics, "stage_mae": _stage_mae(recent_rows)},
            "selected_vs_no_decay_improvement_pct": round(float(improvement_pct), 3),
            "selected_vs_no_decay_mae_difference_pp": round(float(no_decay_mae - winner_mae), 3),
            "selected_vs_recent_only_mae_difference_pp": round(float(winner_mae - recent_mae), 3),
            "bootstrap": _bootstrap_difference(no_decay_rows, winner_rows),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    summary = {
        "winner": winner,
        "half_life_years": CANDIDATES[winner],
        "rolling_mean_mae": summaries[winner]["mean_mae"],
        "no_decay_rolling_mean_mae": summaries["A_no_decay"]["mean_mae"],
        "final_no_decay_mae": no_decay_metrics["MAE"],
        "final_winner_mae": winner_metrics["MAE"],
        "final_recent_only_mae": recent_metrics["MAE"],
        "final_improvement_pct": round(float(improvement_pct), 3),
        "bootstrap": report["final_untouched_holdout"]["bootstrap"],
    }
    print("RECENCY_AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True))
    print(f"AUDIT_FILE={OUT.resolve()}")


if __name__ == "__main__":
    main()
