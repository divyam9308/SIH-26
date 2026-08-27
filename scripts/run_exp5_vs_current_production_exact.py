"""Exact, artifact-free verification of Experiment 5 vs current production.

This reproduces every model-affecting step of current production while skipping
only baseline ablations, SHAP generation, and artifact publication. Current
production means promoted Experiment 12 for cost, retained lifecycle production
for delay/risk. Both production and Exp5 are scored on the same 2022-2025 rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml import monthly_training as trainer
from backend.app.ml.experiments.common_holdout_training_window import audited_feature_contract
from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.trajectory_exp12_v2 import _select_target_features, _usable_features
from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
    training_as_of_invariants,
)
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_BASELINE,
    PRODUCTION_COST_SEED,
    enrich_supervised_for_production,
)
from backend.app.ml.provenance import frame_fingerprint

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "experiments" / "exp5_vs_current_production.json"
REPORT_MD = ROOT / "reports" / "experiments" / "exp5_vs_current_production.md"
TEST_START, TEST_END = 2022, 2025
WINDOWS = [(2001, 2019, 26519), (2001, 2021, 26521)]


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _cohort_fingerprint(frame: pd.DataFrame) -> str:
    keyed = frame[["canonical_project_id", "snapshot_date"]].copy()
    keyed["canonical_project_id"] = keyed.canonical_project_id.astype(str)
    keyed["snapshot_date"] = pd.to_datetime(keyed.snapshot_date, errors="coerce")
    records = [
        [pid, date.isoformat() if pd.notna(date) else None]
        for pid, date in keyed.sort_values(["canonical_project_id", "snapshot_date"]).itertuples(index=False, name=None)
    ]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _production_features(train: pd.DataFrame) -> tuple[list[str], dict]:
    audit = audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by ablation",
            "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by ablation",
        },
    )
    return list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"])), audit


def _rows(test, cost_pred, delay_pred, risk_pred):
    rows = test[[
        "canonical_project_id", "project_name", "snapshot_date", "completion_year",
        "lifecycle_stage", "actual_cost_overrun_percentage", "actual_delay_days",
        "actual_risk", "sample_weight",
    ]].copy()
    rows["predicted_cost_overrun"] = np.asarray(cost_pred)
    rows["predicted_delay_days"] = np.asarray(delay_pred)
    rows["predicted_risk"] = np.asarray(risk_pred)
    rows["cost_error"] = rows.predicted_cost_overrun - rows.actual_cost_overrun_percentage
    rows["delay_error"] = rows.predicted_delay_days - rows.actual_delay_days
    return rows


def _metrics(test, cost_pred, delay_pred, risk_pred):
    return {
        "cost": trainer._regression_metrics(
            test.actual_cost_overrun_percentage, np.asarray(cost_pred), test.sample_weight, test.canonical_project_id
        ),
        "delay": trainer._regression_metrics(
            test.actual_delay_days, np.asarray(delay_pred), test.sample_weight, test.canonical_project_id
        ),
        "risk": trainer._risk_metrics(test.actual_risk, np.asarray(risk_pred), test.sample_weight),
    }


def _production_run(data: pd.DataFrame, start: int, end: int, common_fp: str) -> dict:
    base_train = data[data.completion_year.between(start, end)].copy()
    base_test = data[data.completion_year.between(TEST_START, TEST_END)].copy()
    if not training_as_of_invariants(base_train)["passed"] or not training_as_of_invariants(base_test)["passed"]:
        raise AssertionError("Production base train/test failed as-of invariants")
    overlap = set(base_train.canonical_project_id.dropna()) & set(base_test.canonical_project_id.dropna())
    if overlap:
        raise AssertionError(f"Production train/common-holdout overlap: {len(overlap)}")

    features, audit = _production_features(base_train)
    # Exact production lifecycle fit: this is the same lifecycle variant and
    # seed used by monthly_training.train_window. Its test argument affects only
    # returned metrics, never model/algorithm selection.
    lifecycle_bundle, _, _ = trainer._train_variant(base_train, base_test, features, 26203)

    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train = enriched[enriched.completion_year.between(start, end)].copy()
    test = enriched[enriched.completion_year.between(TEST_START, TEST_END)].copy()
    if _cohort_fingerprint(test) != common_fp:
        raise AssertionError("Promoted production enrichment changed common-holdout membership")

    usable, trajectory_audit = _usable_features(train)
    cost_algorithm = lifecycle_bundle["selected_algorithms"]["cost"]
    cost_added, cost_group, feature_comparisons = _select_target_features(
        train,
        features,
        usable,
        "actual_cost_overrun_percentage",
        cost_algorithm,
        PRODUCTION_COST_SEED,
    )
    cost_features = list(dict.fromkeys(features + cost_added))
    cost_model = trainer._fit_pipeline(
        trainer._regressors(PRODUCTION_COST_SEED)[cost_algorithm],
        train,
        cost_features,
        "actual_cost_overrun_percentage",
    )
    delay_model = lifecycle_bundle["models"]["delay"]
    risk_model = lifecycle_bundle["models"]["risk"]

    cost_pred = cost_model.predict(test[cost_features])
    delay_pred = np.maximum(0, delay_model.predict(test[features]))
    risk_pred = risk_model.predict(test[features])
    metrics = _metrics(test, cost_pred, delay_pred, risk_pred)
    rows = _rows(test, cost_pred, delay_pred, risk_pred)
    return {
        "metrics": metrics,
        "lifecycle_stage_metrics": trainer._stage_metrics(rows),
        "balanced_stage_summary": trainer._balanced_stage_summary(trainer._stage_metrics(rows)),
        "selected_algorithms": lifecycle_bundle["selected_algorithms"],
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "base_features_used": features,
        "cost_features_used": cost_features,
        "delay_features_used": features,
        "cost_trajectory_feature_group": cost_group,
        "cost_trajectory_features": cost_added,
        "internal_cost_trajectory_feature_comparisons": feature_comparisons,
        "trajectory_feature_availability": trajectory_audit,
        "feature_audit": audit,
        "training_projects": int(base_train.canonical_project_id.nunique()),
        "training_snapshots": int(len(base_train)),
        "test_projects": int(test.canonical_project_id.nunique()),
        "test_snapshots": int(len(test)),
        "training_fingerprint": frame_fingerprint(base_train),
        "rows": rows,
    }


def _exp5_run(data: pd.DataFrame, start: int, end: int, seed: int, common_fp: str) -> dict:
    train = data[data.completion_year.between(start, end)].copy()
    test = data[data.completion_year.between(TEST_START, TEST_END)].copy()
    if _cohort_fingerprint(test) != common_fp:
        raise AssertionError("Exp5 common holdout differs from frozen cohort")
    overlap = set(train.canonical_project_id.dropna()) & set(test.canonical_project_id.dropna())
    if overlap:
        raise AssertionError(f"Exp5 train/common-holdout overlap: {len(overlap)}")
    features, audit = audited_feature_contract(train)
    bundle, metrics, rows = trainer._train_variant(train, test, features, seed)
    return {
        "metrics": metrics,
        "lifecycle_stage_metrics": trainer._stage_metrics(rows),
        "balanced_stage_summary": trainer._balanced_stage_summary(trainer._stage_metrics(rows)),
        "selected_algorithms": bundle["selected_algorithms"],
        "features_used": features,
        "feature_count": len(features),
        "feature_audit": audit,
        "seed": seed,
        "training_projects": int(train.canonical_project_id.nunique()),
        "training_snapshots": int(len(train)),
        "test_projects": int(test.canonical_project_id.nunique()),
        "test_snapshots": int(len(test)),
        "training_fingerprint": frame_fingerprint(train),
        "rows": rows,
    }


def _pct(baseline, candidate):
    return round((baseline - candidate) / baseline * 100.0, 4) if baseline else None


def _paired(production_rows: pd.DataFrame, exp_rows: pd.DataFrame) -> dict:
    keys = ["canonical_project_id", "snapshot_date"]
    p = production_rows[keys + [
        "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight", "lifecycle_stage",
        "predicted_cost_overrun", "predicted_delay_days",
    ]].rename(columns={
        "predicted_cost_overrun": "production_cost",
        "predicted_delay_days": "production_delay",
    })
    e = exp_rows[keys + ["predicted_cost_overrun", "predicted_delay_days"]].rename(columns={
        "predicted_cost_overrun": "experiment_cost",
        "predicted_delay_days": "experiment_delay",
    })
    compare = p.merge(e, on=keys, how="inner", validate="one_to_one")
    if len(compare) != len(p) or len(compare) != len(e):
        raise AssertionError("Production and Exp5 prediction rows are not identical")
    return {
        "cost": paired_project_mae_comparison(
            compare,
            actual="actual_cost_overrun_percentage",
            baseline_prediction="production_cost",
            candidate_prediction="experiment_cost",
        ),
        "delay": paired_project_mae_comparison(
            compare,
            actual="actual_delay_days",
            baseline_prediction="production_delay",
            candidate_prediction="experiment_delay",
            seed=26104,
        ),
    }


def _comparison(prod: dict, exp: dict) -> dict:
    pm, em = prod["metrics"], exp["metrics"]
    paired = _paired(prod["rows"], exp["rows"])
    return {
        "cost_mae_absolute_pp": round(pm["cost"]["MAE"] - em["cost"]["MAE"], 3),
        "cost_mae_improvement_percent": _pct(pm["cost"]["MAE"], em["cost"]["MAE"]),
        "delay_mae_absolute_days": round(pm["delay"]["MAE"] - em["delay"]["MAE"], 3),
        "delay_mae_improvement_percent": _pct(pm["delay"]["MAE"], em["delay"]["MAE"]),
        "cost_rmse_improvement_percent": _pct(pm["cost"]["RMSE"], em["cost"]["RMSE"]),
        "delay_rmse_improvement_percent": _pct(pm["delay"]["RMSE"], em["delay"]["RMSE"]),
        "cost_r2_delta": round(em["cost"]["R2"] - pm["cost"]["R2"], 4),
        "delay_r2_delta": round(em["delay"]["R2"] - pm["delay"]["R2"], 4),
        "cost_winner": "exp5" if em["cost"]["MAE"] < pm["cost"]["MAE"] else "production" if em["cost"]["MAE"] > pm["cost"]["MAE"] else "tie",
        "delay_winner": "exp5" if em["delay"]["MAE"] < pm["delay"]["MAE"] else "production" if em["delay"]["MAE"] > pm["delay"]["MAE"] else "tie",
        "paired_project_comparison": paired,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Experiment 5 vs current production — exact fixed 2022–2025 holdout",
        "",
        "Production cost = promoted Experiment 12 trajectory baseline; delay/risk = retained lifecycle production.",
        "",
        "| Train | Model | Cost MAE | Delay MAE | Cost RMSE | Delay RMSE | Cost R² | Delay R² |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window, item in payload["windows"].items():
        for model in ("production", "exp5"):
            m = item[model]["metrics"]
            lines.append(f"| {window.replace('_', '–')} | {model} | {m['cost']['MAE']} | {m['delay']['MAE']} | {m['cost']['RMSE']} | {m['delay']['RMSE']} | {m['cost']['R2']} | {m['delay']['R2']} |")
        c = item["comparison"]
        lines += [
            "",
            f"- **{window.replace('_', '–')} cost:** {c['cost_mae_improvement_percent']}% Exp5 improvement vs production; winner = **{c['cost_winner']}**.",
            f"- **{window.replace('_', '–')} delay:** {c['delay_mae_improvement_percent']}% Exp5 improvement vs production; winner = **{c['delay_winner']}**.",
        ]
    lines += [
        "", "## Controls", "",
        f"- Common holdout: {payload['common_holdout']['projects']} projects / {payload['common_holdout']['snapshots']} snapshots, 2022–2025.",
        f"- Cohort fingerprint: `{payload['common_holdout']['fingerprint']}`.",
        "- Both models are fitted from the same frozen supervised PAIMANA dataset and scored on identical project/snapshot keys.",
        "- Production algorithm selection uses the production 26203/26204 seed contract; promoted Exp12 cost feature-group selection uses seed 26203.",
        "- Exp5 preserves Krish's original 26519/26521 seeds and audited 25-feature contract.",
        "- Delay differences are reported, but they are not a pure architecture effect because Exp5 does not introduce a distinct delay architecture and uses a different seed policy.",
    ]
    return "\n".join(lines) + "\n"


def main():
    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    common = data[data.completion_year.between(TEST_START, TEST_END)].copy()
    if common.empty:
        raise ValueError("Common 2022-2025 holdout is empty")
    common_fp = _cohort_fingerprint(common)
    invariants = training_as_of_invariants(common)
    if not invariants["passed"]:
        raise AssertionError(f"Common holdout failed as-of invariants: {invariants}")

    payload = {
        "comparison": "exp5_vs_current_production",
        "verification_mode": "exact_model_affecting_steps_without_artifact_generation",
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "dataset_fingerprint": frame_fingerprint(data),
        "common_holdout": {
            "period": [TEST_START, TEST_END],
            "projects": int(common.canonical_project_id.nunique()),
            "snapshots": int(len(common)),
            "fingerprint": common_fp,
            "as_of_invariants": invariants,
        },
        "windows": {},
    }
    for start, end, exp_seed in WINDOWS:
        print(f"Fitting current production for {start}-{end}...", flush=True)
        production = _production_run(data, start, end, common_fp)
        print(f"Fitting Experiment 5 for {start}-{end}...", flush=True)
        exp5 = _exp5_run(data, start, end, exp_seed, common_fp)
        if (production["test_projects"], production["test_snapshots"]) != (exp5["test_projects"], exp5["test_snapshots"]):
            raise AssertionError("Production and Exp5 test counts differ")
        payload["windows"][f"{start}_{end}"] = {
            "production": {k: v for k, v in production.items() if k != "rows"},
            "exp5": {k: v for k, v in exp5.items() if k != "rows"},
            "comparison": _comparison(production, exp5),
        }
        print(f"Completed {start}-{end}: {payload['windows'][f'{start}_{end}']['comparison']}", flush=True)

    payload = _safe(payload)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    REPORT_MD.write_text(_markdown(payload))
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
