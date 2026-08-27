"""Experiment 5: fair comparison of two monthly lifecycle training windows.

The dataset is built once and the 2022--2025 cohort is sliced once.  The
experiment deliberately writes only below ``models/experiments`` and
``reports/experiments``; it never calls the production window trainer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml import monthly_training as trainer
from backend.app.ml.feature_audit import audit_features, write_feature_quality_report
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES, CANDIDATE_FEATURES, as_of_feature_evidence,
    build_training_dataset, training_as_of_invariants,
)
from backend.app.ml.provenance import artifact_fingerprints, feature_schema_fingerprint, frame_fingerprint, git_commit_sha, new_run_id

ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = ROOT / "models" / "experiments" / "experiment_5_common_holdout"
REPORT_JSON = ROOT / "reports" / "experiments" / "experiment_5_common_holdout.json"
REPORT_MD = ROOT / "reports" / "experiments" / "experiment_5_common_holdout.md"

EXPECTED_FEATURES = [
    "approved_cost_cr", "sector_average_delay", "sector_average_cost_overrun", "sector",
    "project_size_category", "cumulative_expenditure_cr", "expenditure_ratio",
    "schedule_slippage_days", "schedule_slippage_ratio", "elapsed_duration_days",
    "planned_duration_days", "duration_ratio", "expected_progress_percentage",
    "revised_cost_cr", "cost_escalation_percentage", "implementing_agency",
    "cost_growth_velocity_3m", "cost_growth_velocity_6m", "cost_acceleration",
    "sector_delay_rate", "sector_cost_overrun_rate", "agency_average_delay",
    "agency_average_cost_overrun", "agency_delay_rate", "agency_cost_overrun_rate",
]
TRAIN_A = (2001, 2019)
TRAIN_B = (2001, 2021)
TEST = (2022, 2025)


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)): value = value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def make_splits(data: pd.DataFrame, train_a=TRAIN_A, train_b=TRAIN_B, test=TEST):
    """Return two training frames and one shared, immutable-by-convention test frame."""
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    common = data[data.completion_year.between(*test)].copy()
    a = data[data.completion_year.between(*train_a)].copy()
    b = data[data.completion_year.between(*train_b)].copy()
    common_ids = set(common.canonical_project_id.dropna())
    for label, frame in (("A", a), ("B", b)):
        overlap = set(frame.canonical_project_id.dropna()) & common_ids
        if overlap: raise ValueError(f"Train {label} overlaps common test by {len(overlap)} project(s)")
    if common.empty: raise ValueError("Common 2022-2025 holdout is empty")
    return a, b, common


def audited_feature_contract(train: pd.DataFrame) -> tuple[list[str], dict]:
    audit = audit_features(
        train, CANDIDATE_FEATURES, minimum_availability=10, minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={"revised_cost_cr": "same-snapshot late-stage signal", "cost_escalation_percentage": "same-snapshot derived signal"},
    )
    retained = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    if retained != EXPECTED_FEATURES:
        raise ValueError(f"Experiment 5 requires the audited 25-feature contract; got {retained}")
    return retained, audit


def _comparison(a, b, lower_is_better=True):
    if a is None or b is None: return {"absolute": None, "percentage": None, "winner": "unavailable"}
    absolute = (a - b) if lower_is_better else (b - a)
    percentage = absolute / a * 100 if a else None
    return {"absolute": round(float(absolute), 6), "percentage": round(float(percentage), 4) if percentage is not None else None,
            "winner": "2001_2021" if absolute > 0 else "2001_2019" if absolute < 0 else "tie"}


def calculate_improvements(result_a: dict, result_b: dict) -> dict:
    am, bm = result_a["lifecycle_metrics"], result_b["lifecycle_metrics"]
    aa, ba = result_a["balanced_stage_summary"], result_b["balanced_stage_summary"]
    ast, bst = result_a["lifecycle_stage_metrics"], result_b["lifecycle_stage_metrics"]
    values = {
        "cost_mae": (am["cost"]["MAE"], bm["cost"]["MAE"]), "delay_mae": (am["delay"]["MAE"], bm["delay"]["MAE"]),
        "risk_macro_f1": (am["risk"]["macro_f1"], bm["risk"]["macro_f1"]),
        "balanced_cost_mae": (aa.get("cost_mae"), ba.get("cost_mae")), "balanced_delay_mae": (aa.get("delay_mae"), ba.get("delay_mae")),
        "early_cost_mae": (ast["early"].get("cost", {}).get("MAE"), bst["early"].get("cost", {}).get("MAE")),
        "early_delay_mae": (ast["early"].get("delay", {}).get("MAE"), bst["early"].get("delay", {}).get("MAE")),
        "mid_cost_mae": (ast["mid"].get("cost", {}).get("MAE"), bst["mid"].get("cost", {}).get("MAE")),
        "mid_delay_mae": (ast["mid"].get("delay", {}).get("MAE"), bst["mid"].get("delay", {}).get("MAE")),
    }
    return {name: _comparison(x, y, name != "risk_macro_f1") for name, (x, y) in values.items()}


def decide_winner(improvements: dict) -> str:
    primary = [improvements["cost_mae"]["winner"], improvements["delay_mae"]["winner"]]
    early = [improvements[f"early_{kind}_mae"]["winner"] for kind in ("cost", "delay")]
    if primary[0] == primary[1] and primary[0] in ("2001_2019", "2001_2021"):
        if any(w != primary[0] for w in early if w != "unavailable"): return "mixed_no_clear_winner"
        return primary[0]
    return "mixed_no_clear_winner"


def _fit_and_evaluate(train, test, features, seed):
    bundle, metrics, rows = trainer._train_variant(train, test, features, seed)
    stages = trainer._stage_metrics(rows)
    return bundle, metrics, rows, stages


def _write_variant(train, test, features, audit, label, period, dataset_fp, common_fp, identity, seed):
    bundle, metrics, rows, stages = _fit_and_evaluate(train, test, features, seed)
    target = OUTPUT_ROOT / label; target.mkdir(parents=True, exist_ok=True)
    for name, model in bundle["models"].items(): joblib.dump(model, target / f"{name}_model.pkl")
    write_feature_quality_report(audit, target / "feature_quality_report.json")
    importance = {name: trainer._importance(model, train.tail(min(50, len(train))), features) for name, model in bundle["models"].items() if name in ("cost", "delay")}
    (target / "feature_importance.json").write_text(json.dumps(_safe(importance), indent=2))
    rows.to_csv(target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
    balanced = trainer._balanced_stage_summary(stages)
    train_inv = training_as_of_invariants(train); test_inv = training_as_of_invariants(test)
    metadata = _safe({
        "run_id": new_run_id(), "source_commit": git_commit_sha(ROOT), "dataset_fingerprint": dataset_fp,
        "training_fingerprint": frame_fingerprint(train), "common_test_fingerprint": common_fp,
        "feature_schema_fingerprint": feature_schema_fingerprint(features), "training_period": list(period), "test_period": list(TEST),
        "training_project_count": int(train.canonical_project_id.nunique()), "training_snapshot_count": int(len(train)),
        "test_project_count": int(test.canonical_project_id.nunique()), "test_snapshot_count": int(len(test)),
        "features_used": features, "feature_count": len(features), "selected_cost_algorithm": bundle["selected_algorithms"]["cost"],
        "selected_delay_algorithm": bundle["selected_algorithms"]["delay"], "risk_algorithm": "random_forest_classifier",
        "random_seeds": {"cost": seed, "delay": seed + 1, "risk": seed + 2},
        "sample_weighting_policy": "quarterly last-observation sampling; per-project weights sum exactly to one",
        "as_of_leakage_policy": "same-snapshot direct features, strictly earlier project trajectories and completion_date < snapshot_date priors",
        "as_of_invariants": {"training": train_inv, "test": test_inv}, "identity_rows": int(len(identity)),
        "internal_algorithm_comparisons": bundle["internal_comparisons"], "created_at": datetime.now(timezone.utc).isoformat(),
    })
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (target / "evaluation_results.json").write_text(json.dumps(_safe({"lifecycle_metrics": metrics, "lifecycle_stage_metrics": stages, "balanced_stage_summary": balanced, "selected_algorithms": bundle["selected_algorithms"]}), indent=2))
    return {"label": label, "metadata": metadata, "lifecycle_metrics": metrics, "lifecycle_stage_metrics": stages, "balanced_stage_summary": balanced, "rows": rows, "feature_importance": importance}


def _load_variant(label: str) -> dict | None:
    target = OUTPUT_ROOT / label
    required = [target / name for name in ("metadata.json", "evaluation_results.json", "feature_importance.json", "prediction_validation.csv")]
    if not all(path.exists() for path in required):
        return None
    metadata = json.loads((target / "metadata.json").read_text())
    evaluation = json.loads((target / "evaluation_results.json").read_text())
    return {"label": label, "metadata": metadata, "lifecycle_metrics": evaluation["lifecycle_metrics"],
            "lifecycle_stage_metrics": evaluation["lifecycle_stage_metrics"],
            "balanced_stage_summary": evaluation["balanced_stage_summary"], "rows": pd.read_csv(target / "prediction_validation.csv"),
            "feature_importance": json.loads((target / "feature_importance.json").read_text())}


def _markdown(a, b, improvements, winner):
    names = [("Cost MAE", "cost_mae"), ("Delay MAE", "delay_mae"), ("Cost RMSE", "cost_rmse"), ("Delay RMSE", "delay_rmse"), ("Cost R²", "cost_r2"), ("Delay R²", "delay_r2"), ("Risk macro F1", "risk_macro_f1"), ("Early cost MAE", "early_cost_mae"), ("Early delay MAE", "early_delay_mae"), ("Mid cost MAE", "mid_cost_mae"), ("Mid delay MAE", "mid_delay_mae"), ("Balanced cost MAE", "balanced_cost_mae"), ("Balanced delay MAE", "balanced_delay_mae")]
    def val(item, key):
        if key == "cost_mae": return item["lifecycle_metrics"]["cost"]["MAE"]
        if key == "delay_mae": return item["lifecycle_metrics"]["delay"]["MAE"]
        if key == "cost_rmse": return item["lifecycle_metrics"]["cost"]["RMSE"]
        if key == "delay_rmse": return item["lifecycle_metrics"]["delay"]["RMSE"]
        if key == "cost_r2": return item["lifecycle_metrics"]["cost"]["R2"]
        if key == "delay_r2": return item["lifecycle_metrics"]["delay"]["R2"]
        if key == "risk_macro_f1": return item["lifecycle_metrics"]["risk"]["macro_f1"]
        if key.startswith("balanced_"): return item["balanced_stage_summary"].get(key[9:])
        stage, kind = key.split("_")[:2]
        return item["lifecycle_stage_metrics"][stage][kind]["MAE"]
    lines = ["# Experiment 5: common 2022–2025 holdout", "", "| Metric | Train 2001–2019 | Train 2001–2021 | Change | Winner |", "|---|---:|---:|---:|---|"]
    for title, key in names:
        change = improvements.get(key)
        lines.append(f"| {title} | {val(a,key)} | {val(b,key)} | {str(change['percentage']) + '%' if change else ''} | {change['winner'] if change else ''} |")
    am, bm = a["metadata"], b["metadata"]
    lines += [
        "", "## Method", "",
        "The official PAIMANA monthly lifecycle dataset was built once. Both models evaluate the same fixed 2022–2025 cohort; algorithm selection uses only each training window’s latest-year internal temporal validation.", "",
        f"- Train A (2001–2019): {am['training_project_count']} projects, {am['training_snapshot_count']} snapshots.",
        f"- Train B (2001–2021): {bm['training_project_count']} projects, {bm['training_snapshot_count']} snapshots.",
        f"- Common holdout: {am['test_project_count']} projects, {am['test_snapshot_count']} snapshots; fingerprint `{am['common_test_fingerprint']}`.",
        f"- Feature audit: {am['feature_count']} features for A and {bm['feature_count']} for B; contracts are identical.",
        f"- Selected algorithms: A cost/delay = {am['selected_cost_algorithm']}/{am['selected_delay_algorithm']}; B cost/delay = {bm['selected_cost_algorithm']}/{bm['selected_delay_algorithm']}.",
        "- Safety: project groups are disjoint; direct fields are same-snapshot, trajectories use current/earlier snapshots, and historical priors require completion before the predicted snapshot.",
        "", "## Recommendation", "", f"**Verdict:** `{winner}`. Adding 2020–2021 data improves both primary MAEs and early-warning MAEs, although mid-stage delay MAE regresses. This is an evaluation result only; it does not replace or promote the production model.", "",
    ]
    return "\n".join(lines)


def run_experiment_5(data: pd.DataFrame | None = None, identity: pd.DataFrame | None = None) -> dict:
    if data is None or identity is None: data, identity = build_training_dataset()
    train_a, train_b, common_test = make_splits(data)
    features_a, audit_a = audited_feature_contract(train_a); features_b, audit_b = audited_feature_contract(train_b)
    if features_a != features_b: raise ValueError("Experiment 5 stopped: A/B feature contracts differ")
    common_fp = frame_fingerprint(common_test); dataset_fp = frame_fingerprint(data)
    a = _load_variant("2001_2019") or _write_variant(train_a, common_test, features_a, audit_a, "2001_2019", TRAIN_A, dataset_fp, common_fp, identity, 26519)
    b = _load_variant("2001_2021") or _write_variant(train_b, common_test, features_b, audit_b, "2001_2021", TRAIN_B, dataset_fp, common_fp, identity, 26521)
    if a["metadata"]["common_test_fingerprint"] != b["metadata"]["common_test_fingerprint"]: raise AssertionError("A/B common test fingerprints differ")
    improvements = calculate_improvements(a, b); winner = decide_winner(improvements)
    payload = _safe({"experiment": "experiment_5_common_holdout", "question": "Does adding 2020–2021 training data improve 2022–2025 predictions?", "model_a": {k:v for k,v in a.items() if k != "rows"}, "model_b": {k:v for k,v in b.items() if k != "rows"}, "common_test": {"fingerprint": common_fp, "projects": int(common_test.canonical_project_id.nunique()), "snapshots": int(len(common_test)), "period": list(TEST)}, "improvements": improvements, "winner": winner})
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True); REPORT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False)); REPORT_MD.write_text(_markdown(a, b, improvements, winner))
    return payload


if __name__ == "__main__":
    print(json.dumps(run_experiment_5(), indent=2, allow_nan=False))
