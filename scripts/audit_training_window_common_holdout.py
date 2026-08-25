from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
    training_as_of_invariants,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.provenance import frame_fingerprint, git_commit_sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audits" / "training_window_common_holdout.json"
TARGET = "actual_cost_overrun_percentage"
STAGES = ("early", "mid", "late", "very_late")
SEED = 26203


def safe(v):
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, (np.integer, np.floating)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def feature_audit(train: pd.DataFrame) -> dict:
    return audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "same-snapshot late-stage signal; audited by production ablation",
            "cost_escalation_percentage": "same-snapshot current escalation",
        },
    )


def selected_features(audit: dict) -> list[str]:
    return list(dict.fromkeys(BASELINE_FEATURES + list(audit["features_used"])))


def common_keys(a: pd.DataFrame, b: pd.DataFrame) -> pd.MultiIndex:
    cols = ["canonical_project_id", "snapshot_date"]
    ia = pd.MultiIndex.from_frame(a[cols].astype(str))
    ib = pd.MultiIndex.from_frame(b[cols].astype(str))
    return ia.intersection(ib)


def subset_by_keys(frame: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.MultiIndex.from_frame(out[["canonical_project_id", "snapshot_date"]].astype(str))
    out = out.loc[idx.isin(keys)].copy()
    # Identical project-balanced weighting for the common holdout.
    counts = out.groupby("canonical_project_id")["canonical_project_id"].transform("count").astype(float)
    out["sample_weight_common"] = 1.0 / counts
    return out


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return _regression_metrics(
        frame[TARGET], prediction, frame["sample_weight_common"], frame["canonical_project_id"]
    )


def stage_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    work = frame[["canonical_project_id", "lifecycle_stage", TARGET, "sample_weight_common"]].copy()
    work["prediction"] = prediction
    out = {}
    for stage in STAGES:
        part = work[work.lifecycle_stage.eq(stage)]
        if part.empty:
            out[stage] = {"available": False}
            continue
        # Re-normalize within each stage so projects are balanced inside that stage.
        counts = part.groupby("canonical_project_id")["canonical_project_id"].transform("count").astype(float)
        w = 1.0 / counts
        m = _regression_metrics(part[TARGET], part.prediction, w, part.canonical_project_id)
        out[stage] = {
            "available": True,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "rows": int(len(part)),
            "projects": int(part.canonical_project_id.nunique()),
        }
    return out


def project_diagnostics(frame: pd.DataFrame, pred_long: np.ndarray, pred_recent: np.ndarray) -> dict:
    d = frame[["canonical_project_id", TARGET]].copy()
    d["long_abs"] = np.abs(pred_long - d[TARGET].to_numpy(dtype=float))
    d["recent_abs"] = np.abs(pred_recent - d[TARGET].to_numpy(dtype=float))
    g = d.groupby("canonical_project_id")[["long_abs", "recent_abs"]].mean()
    delta = g.long_abs - g.recent_abs
    tol = 0.1
    return {
        "projects": int(len(g)),
        "recent_better_pct": round(float((delta > tol).mean() * 100), 3),
        "long_better_pct": round(float((delta < -tol).mean() * 100), 3),
        "approx_tied_pct": round(float((delta.abs() <= tol).mean() * 100), 3),
        "median_long_abs_error": round(float(g.long_abs.median()), 3),
        "median_recent_abs_error": round(float(g.recent_abs.median()), 3),
    }


def bootstrap(frame: pd.DataFrame, pred_long: np.ndarray, pred_recent: np.ndarray, samples: int = 1000) -> dict:
    d = frame[["canonical_project_id", TARGET]].copy()
    y = d[TARGET].to_numpy(dtype=float)
    d["long_abs"] = np.abs(pred_long - y)
    d["recent_abs"] = np.abs(pred_recent - y)
    g = d.groupby("canonical_project_id")[["long_abs", "recent_abs"]].mean()
    diff = (g.long_abs - g.recent_abs).to_numpy(dtype=float)
    rng = np.random.default_rng(26103)
    draws = np.empty(samples)
    n = len(diff)
    for i in range(samples):
        draws[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = float(diff.mean())
    if lo > 0:
        verdict = "2018_2021_better"
    elif hi < 0:
        verdict = "2001_2019_better"
    else:
        verdict = "inconclusive"
    return {
        "metric": "project-balanced MAE(2001-2019) minus MAE(2018-2021)",
        "point_difference_pp": round(point, 3),
        "ci95_lower_pp": round(float(lo), 3),
        "ci95_upper_pp": round(float(hi), 3),
        "samples": samples,
        "verdict": verdict,
    }


def run() -> dict:
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    latest = int(data.completion_year.dropna().max())

    long_train, long_test = temporal_project_split(data, 2001, 2019, latest)
    recent_train, recent_test = temporal_project_split(data, 2018, 2021, latest)

    long_audit = feature_audit(long_train)
    recent_audit = feature_audit(recent_train)
    long_features = selected_features(long_audit)
    recent_features = selected_features(recent_audit)
    # Lock the exact same leakage-safe feature set for both models.
    common_features = [f for f in long_features if f in set(recent_features)]

    long_test_2022 = long_test[long_test.completion_year.between(2022, latest)].copy()
    recent_test_2022 = recent_test[recent_test.completion_year.between(2022, latest)].copy()
    keys = common_keys(long_test_2022, recent_test_2022)
    long_common = subset_by_keys(long_test_2022, keys)
    recent_common = subset_by_keys(recent_test_2022, keys)

    sort_cols = ["canonical_project_id", "snapshot_date"]
    long_common = long_common.sort_values(sort_cols).reset_index(drop=True)
    recent_common = recent_common.sort_values(sort_cols).reset_index(drop=True)
    assert len(long_common) == len(recent_common) and len(long_common) > 0
    assert long_common[sort_cols].astype(str).equals(recent_common[sort_cols].astype(str))
    assert np.allclose(long_common[TARGET].to_numpy(float), recent_common[TARGET].to_numpy(float), equal_nan=True)
    assert np.allclose(long_common.sample_weight_common, recent_common.sample_weight_common)

    # Same algorithm family, hyperparameters, and seed. Only training window changes.
    long_model = _fit_pipeline(_regressors(SEED)["extra_trees"], long_train, common_features, TARGET)
    recent_model = _fit_pipeline(_regressors(SEED)["extra_trees"], recent_train, common_features, TARGET)

    pred_long = np.asarray(long_model.predict(long_common[common_features]), dtype=float)
    pred_recent = np.asarray(recent_model.predict(recent_common[common_features]), dtype=float)
    m_long = metrics(long_common, pred_long)
    m_recent = metrics(recent_common, pred_recent)
    improvement = round((m_long["MAE"] - m_recent["MAE"]) / m_long["MAE"] * 100, 3)

    result = {
        "audit": "training_window_common_holdout_cost",
        "source_commit": git_commit_sha(ROOT),
        "dataset_fingerprint": frame_fingerprint(data),
        "dataset_rows": int(len(data)),
        "identity_verified_rows": int(identity.identity_verified.sum()),
        "latest_completion_year": latest,
        "comparison_control": {
            "same_holdout": True,
            "holdout_completion_years": [2022, latest],
            "same_exact_snapshot_keys": True,
            "same_features": True,
            "same_algorithm": "extra_trees",
            "same_hyperparameter_constructor_seed": SEED,
            "same_project_balanced_test_weights": True,
            "only_intended_difference": "training completion-year window",
        },
        "common_holdout": {
            "rows": int(len(long_common)),
            "projects": int(long_common.canonical_project_id.nunique()),
            "features": common_features,
            "feature_count": len(common_features),
        },
        "model_2001_2019": {
            "training_rows": int(len(long_train)),
            "training_projects": int(long_train.canonical_project_id.nunique()),
            "feature_quality_score": long_audit["data_quality_score"],
            "as_of_invariants": training_as_of_invariants(long_train),
            "metrics": m_long,
            "stage_metrics": stage_metrics(long_common, pred_long),
        },
        "model_2018_2021": {
            "training_rows": int(len(recent_train)),
            "training_projects": int(recent_train.canonical_project_id.nunique()),
            "feature_quality_score": recent_audit["data_quality_score"],
            "as_of_invariants": training_as_of_invariants(recent_train),
            "metrics": m_recent,
            "stage_metrics": stage_metrics(recent_common, pred_recent),
        },
        "comparison": {
            "cost_mae_improvement_2018_2021_vs_2001_2019_pct": improvement,
            "winner": "2018_2021" if m_recent["MAE"] < m_long["MAE"] else "2001_2019",
            "project_diagnostics": project_diagnostics(long_common, pred_long, pred_recent),
            "bootstrap": bootstrap(long_common, pred_long, pred_recent),
        },
    }
    return safe(result)


def main():
    result = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    summary = {
        "holdout": result["comparison_control"]["holdout_completion_years"],
        "rows": result["common_holdout"]["rows"],
        "projects": result["common_holdout"]["projects"],
        "features": result["common_holdout"]["feature_count"],
        "mae_2001_2019": result["model_2001_2019"]["metrics"]["MAE"],
        "mae_2018_2021": result["model_2018_2021"]["metrics"]["MAE"],
        "improvement_pct": result["comparison"]["cost_mae_improvement_2018_2021_vs_2001_2019_pct"],
        "winner": result["comparison"]["winner"],
        "bootstrap": result["comparison"]["bootstrap"],
    }
    print("AUDIT_SUMMARY=" + json.dumps(summary, sort_keys=True))
    print("AUDIT_FILE=" + str(OUT))


if __name__ == "__main__":
    main()
