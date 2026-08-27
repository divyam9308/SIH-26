"""Experiment 24: official, dated obstruction-history residual challenger.

The experiment augments PAIMANA with project-specific obstruction evidence from
Government of India / Parliament publications. Every external row has an
explicit publication date and is usable only at snapshots on/after that date.
Unmatched projects keep the production prediction unchanged; absence of a
curated record is never interpreted as proof that no obstruction existed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    enrich_supervised_for_production,
    target_feature_contract,
    train_window_with_promoted_cost,
)

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "reports" / "experiments" / "exp_24"
OBSTRUCTION_HISTORY = ROOT / "data" / "processed" / "exp24_official_obstruction_history.csv"
EXPERIMENT_ID = "exp_24"
EXPERIMENT_NAME = "Official as-of obstruction / delay-reason history"
EXPERIMENT_SCOPE = "cost_delay"
SHRINKAGE_K = 8.0
MAX_COST_CORRECTION = 12.0
MAX_DELAY_CORRECTION = 180.0
MIN_PROMOTION_TRAIN_PROJECTS = 20
MIN_PROMOTION_TEST_PROJECTS = 10

REASON_COLUMNS = [
    "delay_reason", "reason_for_delay", "obstruction_reason", "obstruction",
    "constraints", "issues", "remarks", "project_remarks", "delay_remarks",
]
CATEGORY_PATTERNS = {
    "land_acquisition": r"land acquisition|acquisition of land|compensation|right of use|right of way|\brou\b|\brow\b",
    "clearance": r"environment|forest|wildlife|moef|clearance|permission|consent to operate|tree[- ]?felling|tree cutting",
    "utility_shifting": r"utility shifting|shifting of utilit|electric line|water pipeline|infringing utilit",
    "contractor": r"contractor|concessionaire|mobilisation|mobilization|poor progress|slow progress|insolvency|nclt",
    "funding": r"financial|fund|finance|cash flow|payment|coal linkage|ppa",
    "litigation": r"court|litigation|arbitration|contractual dispute|legal dispute|ngt",
    "law_order": r"law and order|bandh|agitation|protest|violence|security|naxal",
    "geology_weather": r"geolog|soil|landslide|flood|monsoon|rain|weather|terrain|water ingress|seepage",
    "procurement_supply": r"tender|procurement|award|supply|material|equipment|bhel|boiler",
    "rehabilitation": r"rehabilitation|resettlement|relocation|homestead|compensation",
    "design_scope": r"design|scope|descop|plot plan|foundation|alignment",
    "civil_work": r"civil work|civil front|construction work|erection|working hours",
    "force_majeure": r"covid|pandemic|force majeure",
}
CATEGORY_FEATURES = [f"exp24_{name}" for name in CATEGORY_PATTERNS]

_STOPWORDS = {
    "project", "section", "package", "pkg", "stage", "unit", "road", "line",
    "lane", "laning", "new", "construction", "rehabilitation", "upgradation",
    "upgrade", "widening", "with", "from", "to", "of", "the", "and", "in",
    "km", "nh", "two", "four", "six", "state", "scheme", "basis",
}


def categorize_obstruction_reason(value: object) -> dict[str, float]:
    text = "" if value is None or pd.isna(value) else str(value).lower()
    return {name: float(bool(re.search(pattern, text, flags=re.I))) for name, pattern in CATEGORY_PATTERNS.items()}


def reason_text(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Retain the old canonical-column audit helper; numeric status is excluded."""
    present = [column for column in REASON_COLUMNS if column in frame.columns]
    if not present:
        return pd.Series("", index=frame.index, dtype="string"), []
    text = frame[present].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    return text.astype("string"), present


def load_official_obstruction_history(path: Path = OBSTRUCTION_HISTORY) -> pd.DataFrame:
    history = pd.read_csv(path, dtype="string").fillna("")
    history["source_publication_date"] = pd.to_datetime(history["source_publication_date"], errors="raise")
    for name in CATEGORY_PATTERNS:
        history[name] = history["reason_text"].map(lambda value, n=name: categorize_obstruction_reason(value)[n])
    history["source_row_id"] = np.arange(len(history), dtype=int)
    return history


def _normalise(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).lower()
    replacements = {
        "stpp": "super thermal power", "tpp": "thermal power", "ccpp": "combined cycle power",
        "ccgt": "combined cycle power", "hep": "hydro electric", "–": "-", "—": "-", "&": " and ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokens(value: object) -> set[str]:
    return {token for token in _normalise(value).split() if token not in _STOPWORDS and not token.isdigit()}


def _name_score(source: object, candidate: object) -> float:
    a = _normalise(source); b = _normalise(candidate)
    if not a or not b:
        return 0.0
    at = _tokens(a); bt = _tokens(b)
    if not at or not bt:
        return 0.0
    overlap = len(at & bt)
    coverage = overlap / len(at)
    jaccard = overlap / len(at | bt)
    sequence = SequenceMatcher(None, a, b).ratio()
    containment = 1.0 if (a in b or b in a) and overlap >= min(2, len(at)) else 0.0
    return max(containment, 0.62 * coverage + 0.23 * jaccard + 0.15 * sequence)


def match_history_to_projects(frame: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    projects = (
        frame.sort_values("snapshot_date")
        .groupby("canonical_project_id", as_index=False)
        .agg(project_name=("project_name", "first"), state=("state", "first"), sector=("sector", "first"))
    )
    project_rows = projects.to_dict("records")
    matched_rows: list[dict] = []
    audit_rows: list[dict] = []
    for _, source in history.iterrows():
        ranked: list[tuple[float, dict]] = []
        for candidate in project_rows:
            score = _name_score(source.project_name_hint, candidate.get("project_name"))
            if score <= 0:
                continue
            state_hint = _tokens(source.get("state", ""))
            candidate_state = _tokens(candidate.get("state", ""))
            if state_hint and candidate_state and state_hint & candidate_state:
                score += 0.06
            sector_hint = _tokens(source.get("sector_hint", ""))
            candidate_sector = _tokens(candidate.get("sector", ""))
            if sector_hint and candidate_sector and sector_hint & candidate_sector:
                score += 0.03
            ranked.append((min(score, 1.0), candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score = ranked[0][0] if ranked else 0.0
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        accepted = bool(ranked) and best_score >= 0.58 and (best_score >= 0.88 or best_score - second >= 0.06)
        audit = {
            "source_row_id": int(source.source_row_id),
            "project_name_hint": str(source.project_name_hint),
            "best_score": round(float(best_score), 4),
            "second_score": round(float(second), 4),
            "accepted": accepted,
            "canonical_project_id": str(ranked[0][1]["canonical_project_id"]) if accepted else None,
            "matched_project_name": str(ranked[0][1]["project_name"]) if accepted else None,
        }
        audit_rows.append(audit)
        if accepted:
            row = source.to_dict()
            row["canonical_project_id"] = ranked[0][1]["canonical_project_id"]
            row["match_score"] = best_score
            matched_rows.append(row)
    matched = pd.DataFrame(matched_rows)
    audit = {
        "official_source_rows": int(len(history)),
        "matched_source_rows": int(len(matched_rows)),
        "matched_unique_projects": int(matched["canonical_project_id"].nunique()) if not matched.empty else 0,
        "match_rate": float(len(matched_rows) / len(history)) if len(history) else 0.0,
        "matches": audit_rows,
        "matching_policy": "training-target-free project identity match; >=0.58 name score plus uniqueness margin or >=0.88 strong match; state/sector only disambiguate",
    }
    return matched, audit


def add_asof_obstruction_features(frame: pd.DataFrame, matched_history: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["exp24_reason_observed"] = 0.0
    out["exp24_source_count"] = 0.0
    out["exp24_months_since_first_reason"] = np.nan
    out["exp24_months_since_latest_reason"] = np.nan
    for feature in CATEGORY_FEATURES:
        out[feature] = 0.0
    if matched_history.empty:
        return out
    by_project = {str(key): group.sort_values("source_publication_date") for key, group in matched_history.groupby("canonical_project_id")}
    for project_id, indices in out.groupby("canonical_project_id").groups.items():
        records = by_project.get(str(project_id))
        if records is None:
            continue
        for idx in indices:
            snapshot = out.at[idx, "snapshot_date"]
            if pd.isna(snapshot):
                continue
            available = records[records.source_publication_date.le(snapshot)]
            if available.empty:
                continue
            out.at[idx, "exp24_reason_observed"] = 1.0
            out.at[idx, "exp24_source_count"] = float(len(available))
            first = available.source_publication_date.min(); latest = available.source_publication_date.max()
            out.at[idx, "exp24_months_since_first_reason"] = max(0.0, (snapshot - first).days / 30.4375)
            out.at[idx, "exp24_months_since_latest_reason"] = max(0.0, (snapshot - latest).days / 30.4375)
            for name, feature in zip(CATEGORY_PATTERNS, CATEGORY_FEATURES):
                out.at[idx, feature] = float(available[name].max())
    return out


def _project_residual_categories(frame: pd.DataFrame, predictions: np.ndarray, target: str) -> pd.DataFrame:
    rows = frame[["canonical_project_id", target, "exp24_reason_observed", *CATEGORY_FEATURES]].copy()
    rows["prediction"] = np.asarray(predictions, dtype=float)
    rows["residual"] = pd.to_numeric(rows[target], errors="coerce") - rows["prediction"]
    aggregations = {"residual": "mean", "exp24_reason_observed": "max", **{feature: "max" for feature in CATEGORY_FEATURES}}
    projects = rows.groupby("canonical_project_id", as_index=False).agg(aggregations)
    return projects[projects.exp24_reason_observed.gt(0)].copy()


def _category_priors(projects: pd.DataFrame, cap: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for feature in CATEGORY_FEATURES:
        values = projects.loc[projects[feature].gt(0), "residual"].dropna()
        if values.empty:
            continue
        shrunk = float(values.mean()) * len(values) / (len(values) + SHRINKAGE_K)
        result[feature] = float(np.clip(shrunk, -cap, cap))
    return result


def _corrections(frame: pd.DataFrame, priors: dict[str, float], cap: float) -> np.ndarray:
    corrections_out = np.zeros(len(frame), dtype=float)
    for position, (_, row) in enumerate(frame.iterrows()):
        if float(row.get("exp24_reason_observed", 0.0) or 0.0) <= 0:
            continue
        values = [value for feature, value in priors.items() if float(row.get(feature, 0.0) or 0.0) > 0]
        if values:
            corrections_out[position] = float(np.clip(np.mean(values), -cap, cap))
    return corrections_out


def _key(row: pd.Series) -> tuple[str, str]:
    return str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()


def _improvement(base: float, challenger: float) -> float:
    return (base - challenger) / base * 100.0 if base else 0.0


def _decision(cost_gain: float, delay_gain: float, train_covered: int, test_covered: int) -> str:
    coverage_ok = train_covered >= MIN_PROMOTION_TRAIN_PROJECTS and test_covered >= MIN_PROMOTION_TEST_PROJECTS
    return "PROMOTION CANDIDATE" if coverage_ok and cost_gain >= 0 and delay_gain >= 0 and (cost_gain > 0 or delay_gain > 0) else "REGRESSION / DO NOT PROMOTE"


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, **_):
    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    history = load_official_obstruction_history()
    matched, match_audit = match_history_to_projects(enriched, history)
    enriched = add_asof_obstruction_features(enriched, matched)
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    contract = target_feature_contract(production_bundle.get("metadata") or {})
    cost_model = production_bundle["cost"]; delay_model = production_bundle["delay"]

    train_cost_pred = cost_model.predict(train[contract["cost"]])
    train_delay_pred = np.maximum(0, delay_model.predict(train[contract["delay"]]))
    prod_cost_pred = cost_model.predict(test[contract["cost"]])
    prod_delay_pred = np.maximum(0, delay_model.predict(test[contract["delay"]]))

    cost_projects = _project_residual_categories(train, train_cost_pred, "actual_cost_overrun_percentage")
    delay_projects = _project_residual_categories(train, train_delay_pred, "actual_delay_days")
    cost_priors = _category_priors(cost_projects, MAX_COST_CORRECTION)
    delay_priors = _category_priors(delay_projects, MAX_DELAY_CORRECTION)
    cost_corr = _corrections(test, cost_priors, MAX_COST_CORRECTION)
    delay_corr = _corrections(test, delay_priors, MAX_DELAY_CORRECTION)
    exp_cost_pred = prod_cost_pred + cost_corr
    exp_delay_pred = np.maximum(0, prod_delay_pred + delay_corr)

    prod_cost = _regression_metrics(test.actual_cost_overrun_percentage, prod_cost_pred, test.sample_weight, test.canonical_project_id)
    exp_cost = _regression_metrics(test.actual_cost_overrun_percentage, exp_cost_pred, test.sample_weight, test.canonical_project_id)
    prod_delay = _regression_metrics(test.actual_delay_days, prod_delay_pred, test.sample_weight, test.canonical_project_id)
    exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
    cost_gain = _improvement(float(prod_cost["MAE"]), float(exp_cost["MAE"]))
    delay_gain = _improvement(float(prod_delay["MAE"]), float(exp_delay["MAE"]))
    train_covered = int(train.loc[train.exp24_reason_observed.gt(0), "canonical_project_id"].nunique())
    test_covered = int(test.loc[test.exp24_reason_observed.gt(0), "canonical_project_id"].nunique())
    decision = _decision(cost_gain, delay_gain, train_covered, test_covered)

    predictions = {}
    for position, (_, row) in enumerate(test.iterrows()):
        predictions[_key(row)] = {
            "predicted_cost_overrun": float(exp_cost_pred[position]),
            "predicted_delay_days": float(exp_delay_pred[position]),
            "obstruction_cost_correction": float(cost_corr[position]),
            "obstruction_delay_correction": float(delay_corr[position]),
            "obstruction_reason_observed": bool(row.exp24_reason_observed > 0),
        }
    run_id = f"exp24-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    coverage = {
        "official_source_rows": match_audit["official_source_rows"],
        "matched_source_rows": match_audit["matched_source_rows"],
        "matched_unique_projects_all_years": match_audit["matched_unique_projects"],
        "training_projects_with_asof_reason": train_covered,
        "test_projects_with_asof_reason": test_covered,
        "training_reason_snapshot_share": float(train.exp24_reason_observed.mean()),
        "test_reason_snapshot_share": float(test.exp24_reason_observed.mean()),
        "test_nonzero_cost_correction_share": float(np.mean(np.abs(cost_corr) > 1e-12)),
        "test_nonzero_delay_correction_share": float(np.mean(np.abs(delay_corr) > 1e-12)),
        "promotion_minimum_training_projects": MIN_PROMOTION_TRAIN_PROJECTS,
        "promotion_minimum_test_projects": MIN_PROMOTION_TEST_PROJECTS,
    }
    overall = {
        "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": exp_cost["MAE"],
        "cost_improvement_percentage": round(cost_gain, 4), "improvement_percentage": round(cost_gain, 4),
        "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"],
        "delay_improvement_percentage": round(delay_gain, 4),
        "comparison_test_projects": int(test.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(test)),
        "coverage": coverage, "match_audit": match_audit,
        "category_priors": {"cost": cost_priors, "delay": delay_priors},
        "decision": decision,
    }
    selected = dict((production_bundle.get("metadata") or {}).get("selected_algorithms") or production_receipt.get("selected_algorithms") or {})
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "model_role": "experiment", "promotion_allowed": False,
            "decision": decision, "selected_algorithms": selected,
            "metrics": {"cost": exp_cost, "delay": exp_delay},
            "leakage_policy": "External reason records are official project-level publications and are joined by target-free identity matching; a record is usable only when source_publication_date <= snapshot_date. Unmatched snapshots retain production exactly.",
        },
        "overall_comparison": overall,
        "runtime_state": {"predictions": predictions, "comparable": set(predictions)},
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["predictions"]:
        raise ValueError("No Experiment 24 prediction is available for this project snapshot.")
    result = dict(state["predictions"][key])
    result["predicted_cost_overrun"] = round(result["predicted_cost_overrun"], 4)
    result["predicted_delay_days"] = round(max(0.0, result["predicted_delay_days"]), 4)
    return result


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    data, identity = build_training_dataset()
    temp_root = Path(tempfile.mkdtemp(prefix="sih-exp24-"))
    try:
        production = train_window_with_promoted_cost(training_start, training_end, test_end, data=data, identity=identity, artifact_root=temp_root)
        artifact_dir = temp_root / f"{training_start}_{training_end}"
        metadata = production["metadata"]
        bundle = {
            "metadata": metadata,
            "cost": joblib.load(artifact_dir / "cost_model.pkl"),
            "delay": joblib.load(artifact_dir / "delay_model.pkl"),
        }
        receipt = {
            "run_id": metadata.get("run_id"),
            "selected_algorithms": metadata.get("selected_algorithms") or {},
        }
        fitted = fit_experiment(
            data=data, training_start=training_start, training_end=training_end, test_end=test_end,
            production_bundle=bundle, production_receipt=receipt,
        )
        overall = fitted["overall_comparison"]
        report = {
            "experiment": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "scope": EXPERIMENT_SCOPE,
            "run_id": fitted["experiment"]["run_id"],
            "status": "complete",
            "evaluable": True,
            "decision": overall["decision"],
            "scientific_status": "EVALUATED - OFFICIAL DATED AS-OF OBSTRUCTION HISTORY",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_period": [training_start, training_end],
            "testing_period": [training_end + 1, test_end],
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "metrics": {
                "production_cost_mae": overall["production_cost_mae"],
                "experiment_cost_mae": overall["experiment_cost_mae"],
                "cost_improvement_percentage": overall["cost_improvement_percentage"],
                "production_delay_mae": overall["production_delay_mae"],
                "experiment_delay_mae": overall["experiment_delay_mae"],
                "delay_improvement_percentage": overall["delay_improvement_percentage"],
            },
            "coverage": overall["coverage"],
            "match_audit": overall["match_audit"],
            "category_priors": overall["category_priors"],
            "production_changed": False,
            "leakage_policy": fitted["experiment"]["leakage_policy"],
        }
        out = REPORT_ROOT / f"{training_start}_{training_end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
