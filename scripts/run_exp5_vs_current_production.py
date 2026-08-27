"""Controlled Experiment 5 vs current-production comparison.

Current production is the promoted Experiment 12 cost baseline plus the retained
production delay/risk models. Experiment 5 is Krish's original audited
25-feature lifecycle candidate. Both are fitted from one frozen supervised
PAIMANA dataset and scored on exactly the same 2022-2025 snapshots for each
training window (2001-2019 and 2001-2021).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from backend.app.ml import monthly_training as trainer
from backend.app.ml.experiments.common_holdout_training_window import audited_feature_contract
from backend.app.ml.monthly_lifecycle import build_training_dataset, training_as_of_invariants
from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_BASELINE,
    enrich_supervised_for_production,
    target_feature_contract,
    train_window_with_promoted_cost,
)
from backend.app.ml.provenance import frame_fingerprint

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "experiments" / "exp5_vs_current_production.json"
REPORT_MD = ROOT / "reports" / "experiments" / "exp5_vs_current_production.md"
TEST_START = 2022
TEST_END = 2025
WINDOWS = [(2001, 2019, 26519), (2001, 2021, 26521)]


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _cohort_fingerprint(frame: pd.DataFrame) -> str:
    rows = frame[["canonical_project_id", "snapshot_date"]].copy()
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    payload = [
        [str(project_id), timestamp.isoformat() if not pd.isna(timestamp) else None]
        for project_id, timestamp in rows.sort_values(["canonical_project_id", "snapshot_date"]).itertuples(index=False, name=None)
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _metrics_from_predictions(test: pd.DataFrame, predicted_cost, predicted_delay, predicted_risk):
    metrics = {
        "cost": trainer._regression_metrics(
            test.actual_cost_overrun_percentage,
            np.asarray(predicted_cost),
            test.sample_weight,
            test.canonical_project_id,
        ),
        "delay": trainer._regression_metrics(
            test.actual_delay_days,
            np.asarray(predicted_delay),
            test.sample_weight,
            test.canonical_project_id,
        ),
        "risk": trainer._risk_metrics(test.actual_risk, np.asarray(predicted_risk), test.sample_weight),
    }
    rows = test[
        [
            "canonical_project_id",
            "project_name",
            "snapshot_date",
            "completion_year",
            "lifecycle_stage",
            "actual_cost_overrun_percentage",
            "actual_delay_days",
            "actual_risk",
            "sample_weight",
        ]
    ].copy()
    rows["predicted_cost_overrun"] = np.asarray(predicted_cost)
    rows["predicted_delay_days"] = np.asarray(predicted_delay)
    rows["predicted_risk"] = np.asarray(predicted_risk)
    rows["cost_error"] = rows.predicted_cost_overrun - rows.actual_cost_overrun_percentage
    rows["delay_error"] = rows.predicted_delay_days - rows.actual_delay_days
    stages = trainer._stage_metrics(rows)
    balanced = trainer._balanced_stage_summary(stages)
    return metrics, rows, stages, balanced


def _production_run(data: pd.DataFrame, identity: pd.DataFrame, start: int, end: int, common_fp: str):
    with tempfile.TemporaryDirectory(prefix=f"prod_{start}_{end}_") as temp_dir:
        artifact_root = Path(temp_dir)
        result = train_window_with_promoted_cost(
            start,
            end,
            TEST_END,
            data=data,
            identity=identity,
            artifact_root=artifact_root,
        )
        metadata = result["metadata"]
        contract = target_feature_contract(metadata)
        target = artifact_root / f"{start}_{end}"
        cost_model = joblib.load(target / "cost_model.pkl")
        delay_model = joblib.load(target / "delay_model.pkl")
        risk_model = joblib.load(target / "risk_model.pkl")

        enriched = enrich_supervised_for_production(data.copy())
        enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
        test = enriched[enriched.completion_year.between(TEST_START, TEST_END)].copy()
        if _cohort_fingerprint(test) != common_fp:
            raise AssertionError("Production 2022-2025 cohort differs from the frozen common holdout")

        predicted_cost = cost_model.predict(test[contract["cost"]])
        predicted_delay = np.maximum(0, delay_model.predict(test[contract["delay"]]))
        predicted_risk = risk_model.predict(test[contract["risk"]])
        metrics, rows, stages, balanced = _metrics_from_predictions(
            test, predicted_cost, predicted_delay, predicted_risk
        )
        return {
            "metrics": metrics,
            "lifecycle_stage_metrics": stages,
            "balanced_stage_summary": balanced,
            "selected_algorithms": metadata.get("selected_algorithms"),
            "cost_feature_group": metadata.get("cost_trajectory_feature_group"),
            "cost_features_used": contract["cost"],
            "delay_features_used": contract["delay"],
            "risk_features_used": contract["risk"],
            "production_cost_baseline": metadata.get("production_cost_baseline"),
            "run_id": metadata.get("run_id"),
            "training_fingerprint": (metadata.get("provenance") or {}).get("training_fingerprint") or metadata.get("training_fingerprint"),
            "test_projects": int(test.canonical_project_id.nunique()),
            "test_snapshots": int(len(test)),
            "rows": rows,
        }


def _exp5_run(data: pd.DataFrame, start: int, end: int, seed: int, common: pd.DataFrame, common_fp: str):
    train = data[data.completion_year.between(start, end)].copy()
    train_ids = set(train.canonical_project_id.dropna())
    test_ids = set(common.canonical_project_id.dropna())
    overlap = train_ids & test_ids
    if overlap:
        raise AssertionError(f"Experiment 5 train/test project overlap: {len(overlap)}")
    features, audit = audited_feature_contract(train)
    bundle, metrics, rows = trainer._train_variant(train, common, features, seed)
    if _cohort_fingerprint(rows) != common_fp:
        raise AssertionError("Experiment 5 scored rows differ from the frozen common holdout")
    stages = trainer._stage_metrics(rows)
    balanced = trainer._balanced_stage_summary(stages)
    return {
        "metrics": metrics,
        "lifecycle_stage_metrics": stages,
        "balanced_stage_summary": balanced,
        "selected_algorithms": bundle["selected_algorithms"],
        "features_used": features,
        "feature_count": len(features),
        "feature_audit_count": len(audit.get("features") or audit.get("feature_audit") or []),
        "seed": seed,
        "training_fingerprint": frame_fingerprint(train),
        "test_projects": int(common.canonical_project_id.nunique()),
        "test_snapshots": int(len(common)),
        "rows": rows,
    }


def _pct_improvement(production: float, candidate: float):
    return round((production - candidate) / production * 100.0, 4) if production else None


def _window_comparison(production: dict, candidate: dict):
    pm, cm = production["metrics"], candidate["metrics"]
    result = {
        "cost_mae_absolute_pp": round(pm["cost"]["MAE"] - cm["cost"]["MAE"], 3),
        "cost_mae_improvement_percent": _pct_improvement(pm["cost"]["MAE"], cm["cost"]["MAE"]),
        "delay_mae_absolute_days": round(pm["delay"]["MAE"] - cm["delay"]["MAE"], 3),
        "delay_mae_improvement_percent": _pct_improvement(pm["delay"]["MAE"], cm["delay"]["MAE"]),
        "cost_rmse_improvement_percent": _pct_improvement(pm["cost"]["RMSE"], cm["cost"]["RMSE"]),
        "delay_rmse_improvement_percent": _pct_improvement(pm["delay"]["RMSE"], cm["delay"]["RMSE"]),
        "cost_r2_delta": round(cm["cost"]["R2"] - pm["cost"]["R2"], 4),
        "delay_r2_delta": round(cm["delay"]["R2"] - pm["delay"]["R2"], 4),
    }
    result["cost_winner"] = "exp5" if result["cost_mae_absolute_pp"] > 0 else "production" if result["cost_mae_absolute_pp"] < 0 else "tie"
    result["delay_winner"] = "exp5" if result["delay_mae_absolute_days"] > 0 else "production" if result["delay_mae_absolute_days"] < 0 else "tie"
    return result


def _markdown(payload: dict) -> str:
    lines = [
        "# Experiment 5 vs current production — fixed 2022–2025 holdout",
        "",
        "Current production cost is the promoted Experiment 12 trajectory baseline. Delay/risk are the retained production models.",
        "",
        "| Training window | Model | Cost MAE (pp) | Delay MAE (days) | Cost RMSE | Delay RMSE | Cost R² | Delay R² |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in payload["windows"].items():
        for label in ("production", "exp5"):
            metrics = item[label]["metrics"]
            lines.append(
                f"| {key.replace('_', '–')} | {label} | {metrics['cost']['MAE']} | {metrics['delay']['MAE']} | "
                f"{metrics['cost']['RMSE']} | {metrics['delay']['RMSE']} | {metrics['cost']['R2']} | {metrics['delay']['R2']} |"
            )
        comp = item["comparison"]
        lines += [
            "",
            f"- **{key.replace('_', '–')} cost:** Exp5 improvement vs production = **{comp['cost_mae_improvement_percent']}%** ({comp['cost_winner']} wins).",
            f"- **{key.replace('_', '–')} delay:** Exp5 improvement vs production = **{comp['delay_mae_improvement_percent']}%** ({comp['delay_winner']} wins).",
        ]
    lines += [
        "",
        "## Controls",
        "",
        f"- Frozen holdout: {payload['common_holdout']['projects']} projects / {payload['common_holdout']['snapshots']} snapshots, 2022–2025.",
        f"- Cohort fingerprint: `{payload['common_holdout']['fingerprint']}`.",
        "- Production and Exp5 use the same supervised dataset and exact same scored rows.",
        "- Production cost uses the promoted Exp12 feature-selection contract; Exp5 preserves Krish's audited 25-feature implementation.",
        "- Exp5 preserves Krish's original seeds (26519 for 2001–2019, 26521 for 2001–2021). Therefore any delay difference is not a clean architectural effect because Exp5 does not introduce a distinct delay architecture and uses a different seed policy.",
    ]
    return "\n".join(lines) + "\n"


def main():
    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    common = data[data.completion_year.between(TEST_START, TEST_END)].copy()
    if common.empty:
        raise ValueError("Common 2022-2025 holdout is empty")
    common_fp = _cohort_fingerprint(common)
    common_invariants = training_as_of_invariants(common)
    if not common_invariants.get("passed"):
        raise AssertionError(f"Common holdout violates as-of invariants: {common_invariants}")

    payload = {
        "comparison": "exp5_vs_current_production",
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "dataset_fingerprint": frame_fingerprint(data),
        "common_holdout": {
            "period": [TEST_START, TEST_END],
            "projects": int(common.canonical_project_id.nunique()),
            "snapshots": int(len(common)),
            "fingerprint": common_fp,
            "as_of_invariants": common_invariants,
        },
        "windows": {},
    }

    for start, end, seed in WINDOWS:
        production = _production_run(data, identity, start, end, common_fp)
        exp5 = _exp5_run(data, start, end, seed, common, common_fp)
        if production["test_projects"] != exp5["test_projects"] or production["test_snapshots"] != exp5["test_snapshots"]:
            raise AssertionError("Production and Exp5 do not have identical test sample counts")
        payload["windows"][f"{start}_{end}"] = {
            "production": {k: v for k, v in production.items() if k != "rows"},
            "exp5": {k: v for k, v in exp5.items() if k != "rows"},
            "comparison": _window_comparison(production, exp5),
        }

    payload = _safe(payload)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False))
    REPORT_MD.write_text(_markdown(payload))
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
