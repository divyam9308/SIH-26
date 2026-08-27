"""Shared leakage-safe harness for isolated monthly neural sequence experiments.

Every prediction at snapshot t uses only official monthly reports for the same
canonical project with snapshot_date <= t. History length is selected only on
rolling folds inside the training period. The future holdout is never used for
architecture/history selection and production is never modified here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import build_experiment_context, experiment_run_directory, new_experiment_manifest
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.monthly_lifecycle import TRAJECTORIES
from backend.app.ml.monthly_training import _regression_metrics, temporal_project_split
from backend.app.ml.production_cost_baseline import target_feature_contract

MIN_SEQUENCE = 3
HISTORY_VARIANTS: dict[str, int | None] = {"12m": 12, "24m": 24, "36m": 36, "60m": 60, "full": None}
SEQUENCE_FEATURES = [
    "approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "physical_progress",
    "schedule_slippage_days", "planned_duration_days", "expected_progress_percentage",
]
STATIC_NUMERIC = ["approved_cost_cr", "planned_duration_days", "elapsed_duration_days", "duration_ratio"]
STATIC_CATEGORICAL = ["sector", "implementing_agency", "project_size_category"]
TARGETS = ["actual_cost_overrun_percentage", "actual_delay_days"]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    experiment_name: str
    experiment_sequence: int
    implementation_revision: str
    architecture_name: str
    seed: int
    hypothesis: str
    artifact_prefix: str
    selection_epochs: int = 8
    final_epochs: int = 10


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def key_for(row: pd.Series) -> tuple[str, str]:
    return str(row.get("canonical_project_id")), pd.Timestamp(row.get("snapshot_date")).isoformat()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SequenceStore:
    def __init__(self, history: pd.DataFrame):
        frame = history.copy()
        frame["canonical_project_id"] = frame["canonical_project_id"].astype("string")
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"]).sort_values(["canonical_project_id", "snapshot_date"])
        self.features = [name for name in SEQUENCE_FEATURES if name in frame]
        if len(self.features) < 4:
            raise ValueError("Neural sequence experiment requires at least four monthly numeric signals.")
        self.projects: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for project, group in frame.groupby("canonical_project_id", sort=False):
            values = group[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            dates = group["snapshot_date"].astype("int64").to_numpy(np.int64)
            self.projects[str(project)] = (dates, values)

    def raw(self, project: object, stamp: object, max_history: int | None) -> np.ndarray | None:
        item = self.projects.get(str(project))
        date = pd.to_datetime(stamp, errors="coerce")
        if item is None or pd.isna(date):
            return None
        dates, values = item
        end = int(np.searchsorted(dates, np.int64(date.value), side="right"))
        if end < MIN_SEQUENCE:
            return None
        start = 0 if max_history is None else max(0, end - int(max_history))
        return values[start:end].copy()

    def count(self, project: object, stamp: object) -> int:
        raw = self.raw(project, stamp, None)
        return 0 if raw is None else int(len(raw))


@dataclass
class NumericScaler:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray]) -> "NumericScaler":
        parts = [np.asarray(a, dtype=np.float32) for a in arrays if a is not None and len(a)]
        if not parts:
            raise ValueError("No training history is available to fit the sequence scaler.")
        values = np.concatenate(parts, axis=0)
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(values), values, median)
        mean = filled.mean(axis=0).astype(np.float32)
        std = filled.std(axis=0).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        return cls(median, mean, std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        return np.concatenate([(filled - self.mean) / self.std, missing.astype(np.float32)], axis=1).astype(np.float32)

    def to_dict(self) -> dict:
        return {"median": self.median.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class StaticEncoder:
    numeric_median: np.ndarray
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    categories: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> "StaticEncoder":
        matrix = np.column_stack([
            pd.to_numeric(rows.get(name, pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(float)
            for name in STATIC_NUMERIC
        ]).astype(np.float32)
        median = np.nanmedian(matrix, axis=0)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(matrix), matrix, median)
        mean = filled.mean(axis=0).astype(np.float32)
        std = filled.std(axis=0).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        categories = {}
        for name in STATIC_CATEGORICAL:
            values = rows.get(name, pd.Series("", index=rows.index)).astype("string").fillna("")
            unique = sorted({str(v) for v in values if str(v) not in {"", "<NA>"}})
            categories[name] = {value: i + 1 for i, value in enumerate(unique)}
        return cls(median, mean, std, categories)

    def transform_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        nums = np.array([pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0] for name in STATIC_NUMERIC], dtype=np.float32)
        missing = ~np.isfinite(nums)
        filled = np.where(missing, self.numeric_median, nums)
        numeric = np.concatenate([(filled - self.numeric_mean) / self.numeric_std, missing.astype(np.float32)]).astype(np.float32)
        cats = np.array([self.categories[name].get(str(row.get(name, "")), 0) for name in STATIC_CATEGORICAL], dtype=np.int64)
        return numeric, cats

    def cardinalities(self) -> list[int]:
        return [len(self.categories[name]) + 1 for name in STATIC_CATEGORICAL]

    def to_dict(self) -> dict:
        return {"numeric_median": self.numeric_median.tolist(), "numeric_mean": self.numeric_mean.tolist(), "numeric_std": self.numeric_std.tolist(), "categories": self.categories}


@dataclass
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> "TargetScaler":
        y = rows[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        w = pd.to_numeric(rows.sample_weight, errors="coerce").fillna(1.0).to_numpy(np.float32)
        w = np.maximum(w, 0)
        w = w / max(float(w.sum()), 1e-9)
        mean = np.sum(y * w[:, None], axis=0)
        variance = np.sum(((y - mean) ** 2) * w[:, None], axis=0)
        std = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
        return cls(mean.astype(np.float32), std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float32) * self.std + self.mean


class SequenceDataset(Dataset):
    def __init__(self, rows, store, max_history, seq_scaler, static, target):
        self.rows = rows.reset_index(drop=True); self.store = store; self.max_history = max_history
        self.seq_scaler = seq_scaler; self.static = static; self.target = target
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows.iloc[index]
        raw = self.store.raw(row.canonical_project_id, row.snapshot_date, self.max_history)
        if raw is None: raise IndexError("SequenceDataset received a row without sufficient monthly history")
        sequence = torch.from_numpy(self.seq_scaler.transform(raw))
        numeric, cats = self.static.transform_row(row)
        y = np.array([row.actual_cost_overrun_percentage, row.actual_delay_days], dtype=np.float32)
        if self.target is not None: y = self.target.transform(y)
        return sequence, torch.from_numpy(numeric), torch.from_numpy(cats), torch.from_numpy(y), float(row.sample_weight)


def collate_sequences(batch):
    sequences, numeric, cats, y, weights = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True), lengths, torch.stack(numeric), torch.stack(cats), torch.stack(y), torch.tensor(weights, dtype=torch.float32)


def eligible_rows(rows, store):
    mask = [store.count(row.canonical_project_id, row.snapshot_date) >= MIN_SEQUENCE for _, row in rows.iterrows()]
    return rows.loc[mask].copy()


def fit_preprocessors(rows, store):
    latest = {}
    for _, row in rows.iterrows():
        project = str(row.canonical_project_id); stamp = pd.Timestamp(row.snapshot_date)
        latest[project] = max(stamp, latest.get(project, stamp))
    return NumericScaler.fit([store.raw(p, s, None) for p, s in latest.items()]), StaticEncoder.fit(rows), TargetScaler.fit(rows)


def build_embeddings(cardinalities):
    dims = [min(12, max(3, int(math.ceil(math.sqrt(card))))) for card in cardinalities]
    return nn.ModuleList([nn.Embedding(card, dim) for card, dim in zip(cardinalities, dims)]), sum(dims)


def train_model(rows, store, max_history, seq_scaler, static, target, model_builder, epochs, seed):
    seed_everything(seed)
    dataset = SequenceDataset(rows, store, max_history, seq_scaler, static, target)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_sequences, generator=torch.Generator().manual_seed(seed))
    model = model_builder(len(store.features) * 2, static.cardinalities())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.5)
    model.train()
    for _ in range(int(epochs)):
        for seq, lengths, numeric, cats, y, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(seq, lengths, numeric, cats)
            per_row = loss_fn(pred, y).mean(dim=1)
            weights = torch.clamp(weights, min=0)
            loss = torch.sum(per_row * weights) / torch.clamp(weights.sum(), min=1e-9)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
    return model


def predict_model(model, rows, store, max_history, seq_scaler, static, target):
    loader = DataLoader(SequenceDataset(rows, store, max_history, seq_scaler, static, None), batch_size=256, shuffle=False, collate_fn=collate_sequences)
    output = []; model.eval()
    with torch.no_grad():
        for seq, lengths, numeric, cats, _y, _weights in loader: output.append(model(seq, lengths, numeric, cats).cpu().numpy())
    standardized = np.concatenate(output, axis=0) if output else np.empty((0, 2), dtype=np.float32)
    return target.inverse(standardized)


def rolling_folds(train, max_folds=3):
    years = sorted(pd.to_numeric(train.completion_year, errors="coerce").dropna().astype(int).unique())
    folds = []
    for year in reversed(years[1:]):
        fit_years = [y for y in years if y < year]; val_years = [year]
        fitting = train[train.completion_year.isin(fit_years)]; validation = train[train.completion_year.isin(val_years)]
        if fitting.canonical_project_id.nunique() >= 10 and validation.canonical_project_id.nunique() >= 2: folds.append((fit_years, val_years))
        if len(folds) >= max_folds: break
    return list(reversed(folds))


def select_history(train, store, config, model_builder):
    folds = rolling_folds(train)
    if not folds: raise ValueError("Neural history selection requires at least one valid internal temporal fold.")
    records = []; by_variant = {name: [] for name in HISTORY_VARIANTS}
    for fold_index, (fit_years, val_years) in enumerate(folds):
        fitting = eligible_rows(train[train.completion_year.isin(fit_years)].copy(), store)
        validation = eligible_rows(train[train.completion_year.isin(val_years)].copy(), store)
        seq_scaler, static, target = fit_preprocessors(fitting, store)
        for variant_index, (label, max_history) in enumerate(HISTORY_VARIANTS.items()):
            model = train_model(fitting, store, max_history, seq_scaler, static, target, model_builder, config.selection_epochs, config.seed + fold_index * 20 + variant_index)
            pred = predict_model(model, validation, store, max_history, seq_scaler, static, target)
            cost = _regression_metrics(validation.actual_cost_overrun_percentage, pred[:, 0], validation.sample_weight, validation.canonical_project_id)
            delay = _regression_metrics(validation.actual_delay_days, np.maximum(0, pred[:, 1]), validation.sample_weight, validation.canonical_project_id)
            record = {"fold": fold_index + 1, "fitting_years": fit_years, "validation_years": val_years, "history_variant": label, "max_months": max_history, "cost_mae": cost["MAE"], "delay_mae": delay["MAE"], "validation_projects": int(validation.canonical_project_id.nunique()), "validation_snapshots": int(len(validation))}
            records.append(record); by_variant[label].append(record)
    summary = [{"history_variant": label, "max_months": HISTORY_VARIANTS[label], "mean_cost_mae": round(float(np.mean([i["cost_mae"] for i in items])), 4), "mean_delay_mae": round(float(np.mean([i["delay_mae"] for i in items])), 4), "folds": len(items)} for label, items in by_variant.items()]
    selected = {"cost": min(summary, key=lambda x: x["mean_cost_mae"])["history_variant"], "delay": min(summary, key=lambda x: x["mean_delay_mae"])["history_variant"]}
    return selected, [{"fold_results": records, "variant_summary": summary}]


def stage_metrics(rows, prefix):
    result = {}
    for stage in ["early", "mid", "late", "very_late"]:
        part = rows[rows.lifecycle_stage.eq(stage)]
        if part.empty: result[stage] = {"available": False}; continue
        result[stage] = {"available": True, "cost_mae": _regression_metrics(part.actual_cost_overrun_percentage, part[f"{prefix}_cost"].to_numpy(), part.sample_weight, part.canonical_project_id)["MAE"], "delay_mae": _regression_metrics(part.actual_delay_days, part[f"{prefix}_delay"].to_numpy(), part.sample_weight, part.canonical_project_id)["MAE"], "rows": int(len(part)), "projects": int(part.canonical_project_id.nunique())}
    return result


def macro_stage(stages, field):
    values = [float(v[field]) for v in stages.values() if v.get("available") and v.get(field) is not None]
    return round(float(np.mean(values)), 4) if values else None


def fit_sequence_experiment(*, config, model_builder, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None):
    seed_everything(config.seed)
    frame = data.copy(); frame["completion_year"] = pd.to_numeric(frame.completion_year, errors="coerce"); frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date, errors="coerce")
    train, test = temporal_project_split(frame, training_start, training_end, test_end)
    if history is None:
        if not TRAJECTORIES.exists(): raise FileNotFoundError(f"{config.experiment_name} requires paimana_project_trajectories.csv")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    store = SequenceStore(history); train = eligible_rows(train, store); compare = eligible_rows(test, store)
    if train.canonical_project_id.nunique() < 10 or compare.canonical_project_id.nunique() < 2: raise ValueError("Insufficient projects with monthly histories for neural sequence comparison.")
    selected, selection_records = select_history(train, store, config, model_builder)
    seq_scaler, static, target = fit_preprocessors(train, store); fitted = {}
    for index, label in enumerate(sorted(set(selected.values()))): fitted[label] = train_model(train, store, HISTORY_VARIANTS[label], seq_scaler, static, target, model_builder, config.final_epochs, config.seed + 500 + index)
    cost_pred = predict_model(fitted[selected["cost"]], compare, store, HISTORY_VARIANTS[selected["cost"]], seq_scaler, static, target)[:, 0]
    delay_pred = np.maximum(0, predict_model(fitted[selected["delay"]], compare, store, HISTORY_VARIANTS[selected["delay"]], seq_scaler, static, target)[:, 1])
    metadata = dict(production_bundle.get("metadata") or {}); contract = target_feature_contract(metadata)
    compare["production_cost"] = production_bundle["cost"].predict(compare[list(contract["cost"])])
    compare["production_delay"] = np.maximum(0, production_bundle["delay"].predict(compare[list(contract["delay"])]))
    compare["experiment_cost"] = cost_pred; compare["experiment_delay"] = delay_pred
    prod_cost = _regression_metrics(compare.actual_cost_overrun_percentage, compare.production_cost.to_numpy(), compare.sample_weight, compare.canonical_project_id)
    exp_cost = _regression_metrics(compare.actual_cost_overrun_percentage, compare.experiment_cost.to_numpy(), compare.sample_weight, compare.canonical_project_id)
    prod_delay = _regression_metrics(compare.actual_delay_days, compare.production_delay.to_numpy(), compare.sample_weight, compare.canonical_project_id)
    exp_delay = _regression_metrics(compare.actual_delay_days, compare.experiment_delay.to_numpy(), compare.sample_weight, compare.canonical_project_id)
    cost_gain = (prod_cost["MAE"] - exp_cost["MAE"]) / prod_cost["MAE"] * 100 if prod_cost["MAE"] else None; delay_gain = (prod_delay["MAE"] - exp_delay["MAE"]) / prod_delay["MAE"] * 100 if prod_delay["MAE"] else None
    prod_stage = stage_metrics(compare, "production"); exp_stage = stage_metrics(compare, "experiment")
    overall = {"architecture": config.architecture_name, "production_cost_mae": prod_cost["MAE"], "experiment_cost_mae": exp_cost["MAE"], "improvement_percentage": round(float(cost_gain), 4) if cost_gain is not None else None, "production_delay_mae": prod_delay["MAE"], "experiment_delay_mae": exp_delay["MAE"], "delay_improvement_percentage": round(float(delay_gain), 4) if delay_gain is not None else None, "selected_history": selected, "history_length_selection": selection_records, "comparison_test_projects": int(compare.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(compare)), "paired_project_cost_comparison": paired_project_mae_comparison(compare, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost", candidate_prediction="experiment_cost", seed=config.seed + 10), "paired_project_delay_comparison": paired_project_mae_comparison(compare, actual="actual_delay_days", baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=config.seed + 11), "production_stage_metrics": prod_stage, "experiment_stage_metrics": exp_stage, "stage_balanced": {"production_cost_mae": macro_stage(prod_stage, "cost_mae"), "experiment_cost_mae": macro_stage(exp_stage, "cost_mae"), "production_delay_mae": macro_stage(prod_stage, "delay_mae"), "experiment_delay_mae": macro_stage(exp_stage, "delay_mae")}, "scientific_decision": "PENDING_TWO_WINDOW_AUDIT"}
    context = build_experiment_context(experiment_id=config.experiment_id, full_data=frame, train=train, test=compare, features=[f"monthly_sequence:{name}" for name in store.features] + STATIC_NUMERIC + STATIC_CATEGORICAL, training_start=training_start, training_end=training_end, testing_end=test_end, weighting_policy="project-balanced quarterly prediction snapshots; all monthly reports up to each prediction date", baseline_name=metadata.get("production_cost_baseline", "production"))
    manifest = new_experiment_manifest(context=context, name=config.experiment_name, changed_dimension="algorithm", hypothesis=config.hypothesis)
    manifest.update({"scope": "cost_delay", "implementation_revision": config.implementation_revision, "architecture": config.architecture_name, "production_run_id": production_receipt.get("run_id"), "production_cost_baseline": metadata.get("production_cost_baseline"), "history_variants": HISTORY_VARIANTS, "selected_history": selected, "sequence_features": store.features, "static_numeric": STATIC_NUMERIC, "static_categorical": STATIC_CATEGORICAL, "min_sequence_reports": MIN_SEQUENCE, "selection_policy": "up to three rolling completion-year folds inside training; cost and delay select history length independently; future holdout untouched", "leakage_policy": "For prediction snapshot t, SequenceStore exposes only official reports with snapshot_date <= t. Fold preprocessing and category maps are fit on fitting projects only.", "promotion_rule": "No automatic promotion; both standard windows and paired-project evidence must be reviewed separately for cost and delay."})
    run_dir = experiment_run_directory(config.experiment_id, context.window, manifest["run_id"]); run_dir.mkdir(parents=True, exist_ok=False)
    for label, model in fitted.items(): torch.save(model.state_dict(), run_dir / f"{config.artifact_prefix}_{label}.pt")
    (run_dir / "preprocessing.json").write_text(json.dumps(safe_json({"sequence": seq_scaler.to_dict(), "static": static.to_dict(), "target": {"mean": target.mean.tolist(), "std": target.std.tolist()}, "selected_history": selected}), indent=2, allow_nan=False) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(safe_json(manifest), indent=2, allow_nan=False) + "\n"); (run_dir / "evaluation_results.json").write_text(json.dumps(safe_json(overall), indent=2, allow_nan=False) + "\n")
    record_experiment({"experiment_id": config.experiment_id, "name": config.experiment_name, "run_id": manifest["run_id"], "status": "COMPLETED", "decision": "PENDING", "model_role": "experiment", "promotion_allowed": False, "scope": "cost_delay", "window": context.window, "created_at": manifest["created_at"], "production_run_id": production_receipt.get("run_id"), "cost_improvement_percentage": overall["improvement_percentage"], "delay_improvement_percentage": overall["delay_improvement_percentage"], "implementation_revision": config.implementation_revision})
    return {"experiment": {"experiment_id": config.experiment_id, "experiment_name": config.experiment_name, "run_id": manifest["run_id"], "scope": "cost_delay", "decision": "PENDING", "promotion_allowed": False, "implementation_revision": config.implementation_revision, "architecture": config.architecture_name, "selected_history": selected, "metrics": {"cost": exp_cost, "delay": exp_delay}}, "overall_comparison": overall, "runtime_state": {"models": fitted, "selected_history": selected, "store": store, "sequence_scaler": seq_scaler, "static": static, "target": target, "comparable": {key_for(row) for _, row in compare.iterrows()}, "config": config}}


def filter_comparable_rows(frame, state):
    return frame[frame.apply(lambda row: key_for(row) in state["comparable"], axis=1)].copy()


def predict_from_state(row, state):
    if key_for(row) not in state["comparable"]: raise ValueError("No comparable neural monthly sequence is available for this snapshot.")
    one = pd.DataFrame([row]); selected = state["selected_history"]
    cost = predict_model(state["models"][selected["cost"]], one, state["store"], HISTORY_VARIANTS[selected["cost"]], state["sequence_scaler"], state["static"], state["target"])[0, 0]
    delay = predict_model(state["models"][selected["delay"]], one, state["store"], HISTORY_VARIANTS[selected["delay"]], state["sequence_scaler"], state["static"], state["target"])[0, 1]
    config = state["config"]
    return {"predicted_cost_overrun": round(float(cost), 4), "predicted_delay_days": round(max(0.0, float(delay)), 4), "selected_history": selected, "reports_available": state["store"].count(row.canonical_project_id, row.snapshot_date), "implementation_revision": config.implementation_revision, "architecture": config.architecture_name}
