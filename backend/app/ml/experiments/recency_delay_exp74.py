"""Experiment 74: Delay recency adaptation on top of post-PR110 production.

The current production Delay model (Exp61 + promoted U1 residual booster) is the
anchor. This experiment asks whether the *remaining* Delay error is better
explained by recent history than by the full archive.

Candidate policies are pre-registered:
- full-history residual correction control,
- rolling 15/10/7/5-year OOF histories,
- exponential half-life 3/5/8/12-year OOF weighting,
- conservative production/recency blends.

Policy and blend weight are selected only from forward training OOF evidence.
Future holdout outcomes are never used for selection. Cost is copied exactly from
current production.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.nextgen_common import _hash_prod, _prepare, normalize_taxonomy
import backend.app.ml.experiments.nextgen_common as nextgen_common
from backend.app.ml.experiments.prediction_ledger import (
    assert_prediction_ledger_matches_cohort,
    build_prediction_ledger,
    write_experiment_prediction_ledger,
)
from backend.app.ml.monthly_lifecycle import (
    assign_project_balanced_weights,
    build_training_dataset,
)
from backend.app.ml.monthly_training import _json_safe, _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import _production_cost_evaluation_rows
import backend.app.ml.production_exp35_baseline as exp35_production
import backend.app.ml.production_exp61_baseline as exp61_production
import backend.app.ml.production_u1_delay_baseline as u1_production

EXPERIMENT_ID = "exp_74"
EXPERIMENT_SEQUENCE = 74
EXPERIMENT_NAME = "Delay recency weighting, rolling windows, temporal decay, and OOF hybrid"
EXPERIMENT_SCOPE = "delay"
MARKER = "EXP74"
CHANGED_DIMENSION = "training_only_delay_recency_adaptation_of_post_u1_residual_error"

WINDOWS = {
    2019: (2020, 2025),
    2021: (2022, 2025),
    2022: (2023, 2025),
    2023: (2024, 2025),
}
POLICIES = (
    {"name": "full_history_control", "kind": "full"},
    {"name": "rolling_15y", "kind": "rolling", "years": 15},
    {"name": "rolling_10y", "kind": "rolling", "years": 10},
    {"name": "rolling_7y", "kind": "rolling", "years": 7},
    {"name": "rolling_5y", "kind": "rolling", "years": 5},
    {"name": "decay_hl_3y", "kind": "decay", "half_life": 3.0},
    {"name": "decay_hl_5y", "kind": "decay", "half_life": 5.0},
    {"name": "decay_hl_8y", "kind": "decay", "half_life": 8.0},
    {"name": "decay_hl_12y", "kind": "decay", "half_life": 12.0},
)
BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
SELECTION_FOLDS = 5
MIN_POLICY_ROWS = 160
MIN_POLICY_PROJECTS = 25
BOOTSTRAP_SAMPLES = 5000
_PRODUCTION_AFT_SELECTOR = exp35_production._select_aft_calibration_projects


def window_contract(training_end: int) -> tuple[int, int]:
    if training_end not in WINDOWS:
        raise ValueError(f"Exp74 supports only cutoffs {sorted(WINDOWS)}")
    return WINDOWS[training_end]


def select_aft_projects_for_exp74(
    frame: pd.DataFrame,
    limit: int = exp35_production.VERIFIED_AFT_CALIBRATION_PROJECTS,
) -> set[str]:
    """Keep the verified 688 gate when possible; later robustness uses all evidence.

    This is an experiment-local compatibility rule for the 2022/2023 robustness
    windows. It does not modify production modules or the verified 2001-2021 gate.
    """
    try:
        return _PRODUCTION_AFT_SELECTOR(frame, limit)
    except RuntimeError as exc:
        message = str(exc)
        if "AFT evidence" not in message or "cannot form the requested" not in message:
            raise
        required = {"canonical_project_id", "snapshot_date", "planned_completion_date"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("Exp74 AFT selection missing: " + ", ".join(missing))
        eligible = exp35_production.AFTResidualDelayModel._aft_eligible(frame)
        selected = set(
            frame.loc[eligible, "canonical_project_id"]
            .astype("string")
            .dropna()
            .tolist()
        )
        if not selected:
            raise
        return selected


@contextmanager
def _aft_selector_override():
    """Patch every bound selector name only for an isolated Exp74 training call."""
    with patch.object(
        exp35_production,
        "_select_aft_calibration_projects",
        select_aft_projects_for_exp74,
    ), patch.object(
        exp61_production,
        "_select_aft_calibration_projects",
        select_aft_projects_for_exp74,
    ), patch.object(
        u1_production,
        "_select_aft_calibration_projects",
        select_aft_projects_for_exp74,
    ), patch.object(
        nextgen_common,
        "_select_aft_calibration_projects",
        select_aft_projects_for_exp74,
    ):
        yield


def _comparison_cohort(prior_test: pd.DataFrame) -> pd.DataFrame:
    cohort = _production_cost_evaluation_rows(prior_test).copy()
    ids = select_aft_projects_for_exp74(cohort)
    cohort[exp35_production.CALIBRATION_GATE_FEATURE] = (
        cohort["canonical_project_id"].astype("string").isin(ids)
    )
    return assign_project_balanced_weights(cohort)


def _weighted_mae(actual, prediction, weight) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = (
        np.isfinite(actual)
        & np.isfinite(prediction)
        & np.isfinite(weight)
        & (weight >= 0)
    )
    if not mask.any():
        return float("inf")
    if float(weight[mask].sum()) <= 0:
        return float(np.mean(np.abs(actual[mask] - prediction[mask])))
    return float(np.average(np.abs(actual[mask] - prediction[mask]), weights=weight[mask]))


def _current_production_oof(prior_train: pd.DataFrame, production_delay_model) -> pd.DataFrame:
    """Build forward OOF errors for the current post-PR110 U1 production Delay path."""
    if not hasattr(production_delay_model, "base_model"):
        raise TypeError("Exp74 requires the post-PR110 U1 Delay production wrapper")

    base_oof = u1_production._delay_oof_frame(
        prior_train,
        production_delay_model.base_model,
    )
    years = sorted(
        int(x)
        for x in pd.to_numeric(base_oof["oof_year"], errors="coerce").dropna().unique()
    )
    chunks = []
    for year in years[1:]:
        fold_year = pd.to_numeric(base_oof["oof_year"], errors="coerce")
        fit = base_oof.loc[fold_year < year].copy()
        val = base_oof.loc[fold_year == year].copy()
        if len(fit) < MIN_POLICY_ROWS or val.empty:
            continue
        _, _, _, _, correction = u1_production._fit_u1_booster(fit, val)
        anchor = pd.to_numeric(val["production_prediction"], errors="coerce").to_numpy(float)
        prediction = np.maximum(0.0, anchor + correction)
        part = val.copy()
        part["production_prediction"] = prediction
        part["residual"] = (
            pd.to_numeric(part["actual_delay_days"], errors="coerce").to_numpy(float)
            - prediction
        )
        part["oof_year"] = int(year)
        chunks.append(part)

    if len(chunks) < 3:
        raise ValueError("Exp74 requires at least three forward current-production OOF folds")
    return pd.concat(chunks, ignore_index=True)


def _policy_frame(oof: pd.DataFrame, policy: dict, anchor_end: int) -> pd.DataFrame:
    """Return an isolated training view for one pre-registered recency policy."""
    work = oof.copy()
    years = pd.to_numeric(work["oof_year"], errors="coerce")
    kind = policy["kind"]

    if kind == "rolling":
        lower = int(anchor_end) - int(policy["years"]) + 1
        work = work.loc[years >= lower].copy()
    elif kind == "decay":
        age = (float(anchor_end) - years).clip(lower=0.0)
        factor = np.power(0.5, age.to_numpy(float) / float(policy["half_life"]))
        base_weight = pd.to_numeric(work["sample_weight"], errors="coerce").fillna(0.0)
        work["sample_weight"] = base_weight.to_numpy(float) * factor
    elif kind != "full":
        raise ValueError(f"Unknown Exp74 policy kind: {kind}")

    return work


def _selection_years(oof: pd.DataFrame) -> list[int]:
    years = sorted(
        int(x)
        for x in pd.to_numeric(oof["oof_year"], errors="coerce").dropna().unique()
    )
    if len(years) < 3:
        raise ValueError("Exp74 requires multiple forward OOF years")
    return years[-min(SELECTION_FOLDS, len(years) - 1) :]


def _fit_policy_correction(
    fit: pd.DataFrame,
    score: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    booster, medians, features, cap, correction = u1_production._fit_u1_booster(
        fit,
        score,
    )
    del booster
    return correction, {
        "fit_rows": int(len(fit)),
        "fit_projects": int(fit["canonical_project_id"].nunique()),
        "features": list(features),
        "training_medians": medians,
        "correction_cap_abs_residual_q90": float(cap),
    }


def select_recency_policy(oof: pd.DataFrame) -> dict:
    """Select policy and production/recency blend from forward OOF evidence only."""
    selection_years = _selection_years(oof)
    year_series = pd.to_numeric(oof["oof_year"], errors="coerce")
    all_scores = []
    per_policy = {}

    for priority, policy in enumerate(POLICIES):
        parts = []
        fold_details = []
        valid = True
        for year in selection_years:
            historical = oof.loc[year_series < year].copy()
            fit = _policy_frame(historical, policy, year - 1)
            val = oof.loc[year_series == year].copy()
            if (
                len(fit) < MIN_POLICY_ROWS
                or int(fit["canonical_project_id"].nunique()) < MIN_POLICY_PROJECTS
                or val.empty
            ):
                valid = False
                break
            correction, fit_details = _fit_policy_correction(fit, val)
            production = pd.to_numeric(
                val["production_prediction"], errors="coerce"
            ).to_numpy(float)
            recency = np.maximum(0.0, production + correction)
            parts.append(
                pd.DataFrame(
                    {
                        "actual": pd.to_numeric(
                            val["actual_delay_days"], errors="coerce"
                        ).to_numpy(float),
                        "weight": pd.to_numeric(
                            val["sample_weight"], errors="coerce"
                        ).to_numpy(float),
                        "production": production,
                        "recency": recency,
                    }
                )
            )
            fold_details.append(
                {
                    "validation_year": int(year),
                    **fit_details,
                }
            )

        if not valid or len(parts) != len(selection_years):
            per_policy[policy["name"]] = {
                "valid": False,
                "selection_years": selection_years,
            }
            continue

        evidence = pd.concat(parts, ignore_index=True)
        policy_scores = []
        for alpha in BLEND_GRID:
            prediction = (
                (1.0 - float(alpha)) * evidence["production"].to_numpy(float)
                + float(alpha) * evidence["recency"].to_numpy(float)
            )
            mae = _weighted_mae(evidence["actual"], prediction, evidence["weight"])
            record = {
                "policy": policy["name"],
                "policy_priority": int(priority),
                "recency_blend_weight": float(alpha),
                "production_blend_weight": float(1.0 - alpha),
                "mae": float(mae),
                "selection_rows": int(len(evidence)),
            }
            policy_scores.append(record)
            all_scores.append(record)

        best_policy_score = min(
            policy_scores,
            key=lambda x: (x["mae"], x["recency_blend_weight"]),
        )
        per_policy[policy["name"]] = {
            "valid": True,
            "kind": policy["kind"],
            "parameters": {
                k: v for k, v in policy.items() if k not in {"name", "kind"}
            },
            "selection_years": selection_years,
            "folds": fold_details,
            "best_blend": best_policy_score,
            "all_blends": policy_scores,
        }

    finite = [x for x in all_scores if np.isfinite(float(x["mae"]))]
    if not finite:
        raise RuntimeError("Exp74 could not form any valid training-only recency policy")

    selected = min(
        finite,
        key=lambda x: (
            x["mae"],
            x["recency_blend_weight"],
            x["policy_priority"],
        ),
    )
    policy = next(p for p in POLICIES if p["name"] == selected["policy"])
    return {
        "selected_policy": dict(policy),
        "selected_recency_blend_weight": float(selected["recency_blend_weight"]),
        "selected_production_blend_weight": float(selected["production_blend_weight"]),
        "selected_meta_oof_mae": float(selected["mae"]),
        "selection_years": selection_years,
        "selection_folds": int(len(selection_years)),
        "blend_grid": [float(x) for x in BLEND_GRID],
        "policies": per_policy,
        "holdout_used_for_selection": False,
        "selection_source": "forward OOF residuals of current post-PR110 production Delay",
    }


def _metric(frame: pd.DataFrame, actual: str, prediction: np.ndarray) -> float:
    return float(
        _regression_metrics(
            frame[actual],
            prediction,
            frame["sample_weight"],
            frame["canonical_project_id"],
        )["MAE"]
    )


def _gain(base: float, candidate: float) -> float:
    return (float(base) - float(candidate)) / float(base) * 100.0 if float(base) else 0.0


def _persist(
    *,
    cohort: pd.DataFrame,
    production_cost: np.ndarray,
    experiment_cost: np.ndarray,
    production_delay: np.ndarray,
    experiment_delay: np.ndarray,
    details: dict,
    training_end: int,
) -> dict:
    if not np.array_equal(production_cost, experiment_cost):
        raise AssertionError("Exp74 modified Cost predictions")

    production_cost_mae = _metric(
        cohort, "actual_cost_overrun_percentage", production_cost
    )
    experiment_cost_mae = _metric(
        cohort, "actual_cost_overrun_percentage", experiment_cost
    )
    production_delay_mae = _metric(cohort, "actual_delay_days", production_delay)
    experiment_delay_mae = _metric(cohort, "actual_delay_days", experiment_delay)
    delay_gain = _gain(production_delay_mae, experiment_delay_mae)
    verdict = "PROMOTION CANDIDATE" if delay_gain > 0 else "DO NOT PROMOTE"

    scored = cohort.copy()
    scored["production_cost_prediction"] = production_cost
    scored["experiment_cost_prediction"] = experiment_cost
    scored["production_delay_prediction"] = production_delay
    scored["experiment_delay_prediction"] = experiment_delay

    cost_bootstrap = paired_project_mae_comparison(
        scored,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost_prediction",
        candidate_prediction="experiment_cost_prediction",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=74001 + int(training_end),
    )
    delay_bootstrap = paired_project_mae_comparison(
        scored,
        actual="actual_delay_days",
        baseline_prediction="production_delay_prediction",
        candidate_prediction="experiment_delay_prediction",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=74002 + int(training_end),
    )

    window = f"2001_{training_end}"
    extras = [
        x
        for x in [
            "completion_year",
            "lifecycle_stage",
            "sector",
            "implementing_agency",
            "state",
            "project_size_category",
            "approved_cost_cr",
            "cost_escalation_percentage",
            "schedule_slippage_days",
            "duration_ratio",
        ]
        if x in scored
    ]
    ledger = build_prediction_ledger(
        scored,
        experiment_id=EXPERIMENT_ID,
        window=window,
        production_cost_prediction=production_cost,
        experiment_cost_prediction=experiment_cost,
        production_delay_prediction=production_delay,
        experiment_delay_prediction=experiment_delay,
        extra_columns=extras,
    )
    assert_prediction_ledger_matches_cohort(ledger, cohort)
    run_id = f"{EXPERIMENT_ID}-{uuid.uuid4().hex[:10]}"
    saved = write_experiment_prediction_ledger(
        ledger,
        experiment_id=EXPERIMENT_ID,
        window=window,
        run_id=run_id,
        extra_manifest={
            "primary_scope": EXPERIMENT_SCOPE,
            "changed_dimension": CHANGED_DIMENSION,
            "execution_verdict": "EXECUTION VALID",
            "scientific_verdict": verdict,
            "decision_role": "primary" if training_end == 2021 else "robustness",
            "holdout_used_for_selection": False,
        },
    )
    evidence_path = Path(saved["ledger_path"]).parent / "experiment_evidence.json"
    evidence_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "details": details,
                    "cost_bootstrap": cost_bootstrap,
                    "delay_bootstrap": delay_bootstrap,
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    overall = {
        "production_cost_mae": production_cost_mae,
        "experiment_cost_mae": experiment_cost_mae,
        "cost_improvement_percentage": 0.0,
        "production_delay_mae": production_delay_mae,
        "experiment_delay_mae": experiment_delay_mae,
        "delay_improvement_percentage": round(delay_gain, 6),
        "comparison_test_projects": int(cohort["canonical_project_id"].nunique()),
        "comparison_test_snapshots": int(len(cohort)),
        "paired_project_bootstrap_cost": cost_bootstrap,
        "paired_project_bootstrap_delay": delay_bootstrap,
        "execution_verdict": "EXECUTION VALID",
        "scientific_verdict": verdict,
    }
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "changed_dimension": CHANGED_DIMENSION,
            "run_id": run_id,
            "promotion_allowed": False,
            "execution_verdict": "EXECUTION VALID",
            "scientific_verdict": verdict,
            "ledger_path": str(saved["ledger_path"]),
            "ledger_manifest_path": str(saved["manifest_path"]),
            "details": _json_safe(details),
        },
        "overall_comparison": overall,
    }


def fit_experiment(
    *,
    data: pd.DataFrame,
    production_bundle: dict,
    training_start: int,
    training_end: int,
    test_end: int,
    **_,
) -> dict:
    prepared = normalize_taxonomy(_prepare(data))
    train, test = temporal_project_split(
        prepared,
        training_start,
        training_end,
        test_end,
    )
    prior_train, prior_test, _ = exp61_production._build_temporal_delay_priors(
        train,
        test,
    )
    cohort = _comparison_cohort(prior_test)

    cost_model = production_bundle["cost"]
    delay_model = production_bundle["delay"]
    production_cost = np.asarray(cost_model.predict(cohort), dtype=float)
    production_delay = np.maximum(
        0.0,
        np.asarray(delay_model.predict(cohort), dtype=float),
    )

    oof = _current_production_oof(prior_train, delay_model)
    selection = select_recency_policy(oof)
    selected_policy = selection["selected_policy"]
    final_fit = _policy_frame(oof, selected_policy, training_end)
    if (
        len(final_fit) < MIN_POLICY_ROWS
        or int(final_fit["canonical_project_id"].nunique()) < MIN_POLICY_PROJECTS
    ):
        raise RuntimeError("Selected Exp74 recency policy has insufficient final fit evidence")

    score = cohort.copy()
    score["production_prediction"] = production_delay
    correction, final_fit_details = _fit_policy_correction(final_fit, score)
    recency_prediction = np.maximum(0.0, production_delay + correction)
    alpha = float(selection["selected_recency_blend_weight"])
    experiment_delay = np.maximum(
        0.0,
        (1.0 - alpha) * production_delay + alpha * recency_prediction,
    )

    details = {
        "baseline": "current production after PR #110 (Exp61 + promoted U1 Delay)",
        "candidate_family": "post-production Delay residual recency adaptation",
        "full_history_control": True,
        "rolling_windows_years": [15, 10, 7, 5],
        "temporal_decay_half_lives_years": [3, 5, 8, 12],
        "hybrid_blend": "production + selected recency-corrected prediction",
        "selection": selection,
        "final_fit": final_fit_details,
        "current_production_oof_rows": int(len(oof)),
        "current_production_oof_projects": int(oof["canonical_project_id"].nunique()),
        "cost_predictions_identical": True,
        "holdout_used_for_policy_selection": False,
        "holdout_used_for_blend_selection": False,
        "later_cutoffs_are_robustness_only": training_end in (2022, 2023),
        "aft_gate_policy": (
            "verified_688_when_available_else_all_available_aft_evidence_projects"
        ),
    }

    return _persist(
        cohort=cohort,
        production_cost=production_cost,
        experiment_cost=production_cost.copy(),
        production_delay=production_delay,
        experiment_delay=experiment_delay,
        details=details,
        training_end=training_end,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2001)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_test_start, expected_test_end = window_contract(args.end)
    if args.start != 2001 or args.test_end != expected_test_end:
        raise ValueError("Exp74 requires start=2001 and test_end=2025")

    before = _hash_prod()
    data, identity = build_training_dataset()
    with tempfile.TemporaryDirectory(prefix=f"{EXPERIMENT_ID}-{args.end}-") as td:
        root = Path(td) / "production"
        with _aft_selector_override():
            receipt = u1_production.train_window_with_promoted_cost_and_delay(
                args.start,
                args.end,
                args.test_end,
                data=data,
                identity=identity,
                artifact_root=root,
            )
            target = root / f"{args.start}_{args.end}"
            bundle = {
                "cost": joblib.load(target / "cost_model.pkl"),
                "delay": joblib.load(target / "delay_model.pkl"),
                "metadata": json.loads((target / "metadata.json").read_text()),
            }
            fitted = fit_experiment(
                data=data,
                production_bundle=bundle,
                production_receipt=receipt,
                training_start=args.start,
                training_end=args.end,
                test_end=args.test_end,
            )

    if before != _hash_prod():
        raise AssertionError("Exp74 modified tracked production artifacts")

    overall = fitted["overall_comparison"]
    if args.end == 2021 and (
        int(overall["comparison_test_projects"]) != 721
        or int(overall["comparison_test_snapshots"]) != 11200
    ):
        raise RuntimeError("Exp74 verified 2001-2021 cohort changed")

    payload = {
        "window": f"{args.start}_{args.end}",
        "test_start": expected_test_start,
        "test_end": expected_test_end,
        "assumed_production": "post-PR110 U1 Delay production",
        "production": receipt,
        "experiment": fitted["experiment"],
        "overall_comparison": overall,
        "production_artifacts_untouched": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    )

    prefix = f"{MARKER}_{args.start}_{args.end}"
    print(f"{prefix}_PRODUCTION_COST_MAE={overall['production_cost_mae']}")
    print(f"{prefix}_EXPERIMENT_COST_MAE={overall['experiment_cost_mae']}")
    print(f"{prefix}_PRODUCTION_DELAY_MAE={overall['production_delay_mae']}")
    print(f"{prefix}_EXPERIMENT_DELAY_MAE={overall['experiment_delay_mae']}")
    print(
        f"{prefix}_DELAY_IMPROVEMENT_PERCENT="
        f"{overall['delay_improvement_percentage']}"
    )
    selection = fitted["experiment"]["details"]["selection"]
    print(f"{MARKER}_SELECTED_POLICY={selection['selected_policy']['name']}")
    print(
        f"{MARKER}_SELECTED_RECENCY_BLEND_WEIGHT="
        f"{selection['selected_recency_blend_weight']}"
    )
    print(f"{MARKER}_SELECTION_YEARS={selection['selection_years']}")
    print(f"{MARKER}_EXECUTION_VERDICT={overall['execution_verdict']}")
    print(f"{MARKER}_SCIENTIFIC_VERDICT={overall['scientific_verdict']}")


if __name__ == "__main__":
    main()
