"""Experiment 21: leakage-safe project scope semantics and complexity features.

Production is retrained unchanged first. The challenger then keeps the exact
production-selected regressors and adds only information available from the
project name/scope at the prediction snapshot. Text representation is fitted on
training project names only; future holdout names never fit the vocabulary/SVD.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_SEED,
    enrich_supervised_for_production,
    target_feature_contract,
    train_window_with_promoted_cost,
)

ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "reports" / "experiments" / "exp_21"
EXPERIMENT_ID = "exp_21"
EXPERIMENT_NAME = "Project scope semantics + engineering complexity"
EXPERIMENT_SCOPE = "cost_delay"
STRUCTURED_FEATURES = [
    "exp21_scope_chars", "exp21_scope_tokens", "exp21_numeric_tokens",
    "exp21_phase_number", "exp21_length_km", "exp21_capacity_mw",
    "exp21_tunnel", "exp21_bridge", "exp21_bypass", "exp21_pipeline",
    "exp21_transmission", "exp21_hydro", "exp21_rail", "exp21_port",
    "exp21_highway", "exp21_expansion", "exp21_redevelopment", "exp21_greenfield",
]


def _numeric_extract(text: pd.Series, pattern: str) -> pd.Series:
    return pd.to_numeric(text.str.extract(pattern, expand=False), errors="coerce")


def add_structured_scope_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = result.get("project_name", pd.Series("", index=result.index)).fillna("").astype(str)
    text = raw.str.lower()
    result["exp21_scope_chars"] = raw.str.len().astype(float)
    result["exp21_scope_tokens"] = raw.str.findall(r"[A-Za-z0-9]+", flags=re.I).str.len().astype(float)
    result["exp21_numeric_tokens"] = raw.str.findall(r"\d+(?:\.\d+)?").str.len().astype(float)
    result["exp21_phase_number"] = _numeric_extract(text, r"(?:phase|stage|ph\.?)[\s-]*(?:no\.?\s*)?(\d+)")
    result["exp21_length_km"] = _numeric_extract(text, r"(\d+(?:\.\d+)?)\s*(?:km|kms|kilomet(?:er|re)s?)\b")
    result["exp21_capacity_mw"] = _numeric_extract(text, r"(\d+(?:\.\d+)?)\s*mw\b")
    keywords = {
        "exp21_tunnel": r"\btunnel\b",
        "exp21_bridge": r"\bbridge|bridges|flyover\b",
        "exp21_bypass": r"\bbypass\b",
        "exp21_pipeline": r"\bpipeline\b",
        "exp21_transmission": r"\btransmission\b",
        "exp21_hydro": r"\bhydel\b|\bhydro\b|\bhep\b",
        "exp21_rail": r"\brail(?:way)?\b|\bmetro\b|\bnew line\b|\bgauge conversion\b",
        "exp21_port": r"\bport\b|\bharbour\b|\bharbor\b",
        "exp21_highway": r"\bhighway\b|\bnh[- ]?\d+\b|\bfour[- ]lan|\bsix[- ]lan",
        "exp21_expansion": r"\bexpansion\b|\baugmentation\b",
        "exp21_redevelopment": r"\bredevelopment\b|\bmodernisation\b|\bmodernization\b",
        "exp21_greenfield": r"\bgreenfield\b",
    }
    for column, pattern in keywords.items():
        result[column] = text.str.contains(pattern, regex=True, na=False).astype(float)
    return result


def add_semantic_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    train_out = add_structured_scope_features(train)
    test_out = add_structured_scope_features(test)
    train_text = train_out.project_name.fillna("").astype(str)
    test_text = test_out.project_name.fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=1800,
        sublinear_tf=True, lowercase=True,
    )
    matrix = vectorizer.fit_transform(train_text)
    semantic_features: list[str] = []
    explained = 0.0
    if matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
        n_components = min(16, matrix.shape[0] - 1, matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=26221)
        train_sem = svd.fit_transform(matrix)
        test_sem = svd.transform(vectorizer.transform(test_text))
        semantic_features = [f"exp21_scope_sem_{i:02d}" for i in range(n_components)]
        for i, column in enumerate(semantic_features):
            train_out[column] = train_sem[:, i]
            test_out[column] = test_sem[:, i]
        explained = float(np.sum(svd.explained_variance_ratio_))
    audit = {
        "vectorizer": "training-only character TF-IDF 3-5 grams",
        "vocabulary_size": int(matrix.shape[1]),
        "semantic_components": len(semantic_features),
        "svd_explained_variance_ratio_sum": explained,
        "holdout_used_to_fit_text_representation": False,
    }
    return train_out, test_out, STRUCTURED_FEATURES + semantic_features, audit


def _improvement(base: float, challenger: float) -> float:
    return (base - challenger) / base * 100.0 if base else 0.0


def _decision(cost_improvement: float, delay_improvement: float) -> str:
    return "PROMOTION CANDIDATE" if cost_improvement >= 0 and delay_improvement >= 0 and (cost_improvement > 0 or delay_improvement > 0) else "REGRESSION / DO NOT PROMOTE"


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    data, identity = build_training_dataset()
    temp_root = Path(tempfile.mkdtemp(prefix="sih-exp21-"))
    run_id = f"exp21-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    try:
        production = train_window_with_promoted_cost(training_start, training_end, test_end, data=data, identity=identity, artifact_root=temp_root)
        artifact_dir = temp_root / f"{training_start}_{training_end}"
        metadata = production["metadata"]
        contract = target_feature_contract(metadata)
        selected = metadata["selected_algorithms"]

        enriched = enrich_supervised_for_production(data.copy())
        enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
        train, test = temporal_project_split(enriched, training_start, training_end, test_end)

        prod_cost = joblib.load(artifact_dir / "cost_model.pkl")
        prod_delay = joblib.load(artifact_dir / "delay_model.pkl")
        base_cost_pred = prod_cost.predict(test[contract["cost"]])
        base_delay_pred = np.maximum(0, prod_delay.predict(test[contract["delay"]]))

        train_aug, test_aug, added, text_audit = add_semantic_features(train, test)
        cost_features = list(dict.fromkeys(contract["cost"] + added))
        delay_features = list(dict.fromkeys(contract["delay"] + added))
        cost_model = _fit_pipeline(_regressors(PRODUCTION_COST_SEED)[selected["cost"]], train_aug, cost_features, "actual_cost_overrun_percentage")
        delay_model = _fit_pipeline(_regressors(26204)[selected["delay"]], train_aug, delay_features, "actual_delay_days")
        exp_cost_pred = cost_model.predict(test_aug[cost_features])
        exp_delay_pred = np.maximum(0, delay_model.predict(test_aug[delay_features]))

        base_cost = _regression_metrics(test.actual_cost_overrun_percentage, base_cost_pred, test.sample_weight, test.canonical_project_id)
        exp_cost = _regression_metrics(test.actual_cost_overrun_percentage, exp_cost_pred, test.sample_weight, test.canonical_project_id)
        base_delay = _regression_metrics(test.actual_delay_days, base_delay_pred, test.sample_weight, test.canonical_project_id)
        exp_delay = _regression_metrics(test.actual_delay_days, exp_delay_pred, test.sample_weight, test.canonical_project_id)
        cost_gain = _improvement(float(base_cost["MAE"]), float(exp_cost["MAE"]))
        delay_gain = _improvement(float(base_delay["MAE"]), float(exp_delay["MAE"]))

        report = {
            "experiment": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "scope": EXPERIMENT_SCOPE,
            "run_id": run_id, "status": "complete", "decision": _decision(cost_gain, delay_gain),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_period": [training_start, training_end], "testing_period": [training_end + 1, test_end],
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "production_selected_algorithms": selected,
            "added_features": added, "text_audit": text_audit,
            "metrics": {
                "production_cost_mae": float(base_cost["MAE"]), "experiment_cost_mae": float(exp_cost["MAE"]),
                "cost_improvement_percentage": cost_gain,
                "production_delay_mae": float(base_delay["MAE"]), "experiment_delay_mae": float(exp_delay["MAE"]),
                "delay_improvement_percentage": delay_gain,
                "production_cost": base_cost, "experiment_cost": exp_cost,
                "production_delay": base_delay, "experiment_delay": exp_delay,
            },
            "coverage": {"training_projects": int(train.canonical_project_id.nunique()), "test_projects": int(test.canonical_project_id.nunique()), "test_snapshots": int(len(test))},
            "leakage_policy": "Project-name structured features use same-snapshot text. TF-IDF vocabulary and SVD are fitted only on training-window project names. Production model family and temporal cohort are unchanged.",
            "production_changed": False,
        }
        out = REPORT_ROOT / f"{training_start}_{training_end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        validation = test[["canonical_project_id", "project_name", "snapshot_date", "completion_year", "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight"]].copy()
        validation["production_cost_prediction"] = base_cost_pred
        validation["experiment_cost_prediction"] = exp_cost_pred
        validation["production_delay_prediction"] = base_delay_pred
        validation["experiment_delay_prediction"] = exp_delay_pred
        validation.to_csv(out / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
