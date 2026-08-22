"""Experiment 2: hybrid cost-regime classification + regression.

This module is intentionally isolated from production. It reads the registered
2001-2017 production baseline, official completed-project outcomes, and the same
five-feature contract; all generated artifacts live under experiment paths.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)

from backend.app.ml.real_time_windows import (
    CAT_FEATURE_INDICES,
    FEATURES,
    MODELS,
    REPORTS,
    _algorithm_regressor,
    _fit_regressor,
    _predict_regressor,
    _safe_mape,
    add_leave_one_out_training_priors,
    apply_historical_priors,
    apply_sector_correction,
    historical_prior_maps,
    labelled,
    outcome_data,
    window_for,
)

BASELINE_KEY = "2001_2017"
EXPERIMENT_NAME = "hybrid_cost_regime"
EXPERIMENT_DIR = MODELS / "experiments" / EXPERIMENT_NAME
EXPERIMENT_REPORTS = REPORTS / "experiments"
SEED = 26103
MIN_EXPERT_SAMPLES = 12
REGIMES = ["COST_SAVING", "LOW", "MEDIUM", "HIGH", "EXTREME"]
REGIME_BOUNDS = {
    "COST_SAVING": (-90.0, 0.0),
    "LOW": (0.0, 20.0),
    "MEDIUM": (20.0, 100.0),
    "HIGH": (100.0, 200.0),
    "EXTREME": (200.0, 1000.0),
}


def cost_regime(value: float) -> str:
    """Map a historical outcome to a pre-declared cost regime."""
    value = float(value)
    if value <= 0:
        return "COST_SAVING"
    if value <= 20:
        return "LOW"
    if value <= 100:
        return "MEDIUM"
    if value <= 200:
        return "HIGH"
    return "EXTREME"


def regime_series(values) -> pd.Series:
    return pd.Series(values).astype(float).map(cost_regime)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _regime_distribution(labels: pd.Series) -> dict:
    total = max(1, len(labels))
    return {
        regime: {
            "count": int((labels == regime).sum()),
            "percentage": round(float((labels == regime).sum()) / total * 100.0, 2),
            "expert_strategy": "regime-specific CatBoost" if int((labels == regime).sum()) >= MIN_EXPERT_SAMPLES else "global fallback clipped to regime bounds",
        }
        for regime in REGIMES
    }


def _classifier(seed: int = SEED) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=6.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )


def _fit_classifier(frame: pd.DataFrame, seed: int = SEED) -> CatBoostClassifier:
    labels = regime_series(frame.actual_cost_overrun_percentage).to_numpy()
    model = _classifier(seed)
    model.fit(frame[FEATURES], labels, cat_features=CAT_FEATURE_INDICES)
    return model


def _ordered_probabilities(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = [str(item) for item in model.classes_]
    matrix = np.zeros((len(X), len(REGIMES)), dtype=float)
    for index, regime in enumerate(REGIMES):
        if regime in classes:
            matrix[:, index] = raw[:, classes.index(regime)]
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return matrix / totals


def _fit_global(frame: pd.DataFrame, seed: int = SEED):
    model = _algorithm_regressor("catboost_mae_d4", seed)
    _fit_regressor(model, frame[FEATURES], frame.actual_cost_overrun_percentage, delay_target=False)
    return model


def _fit_experts(frame: pd.DataFrame) -> tuple[dict, dict]:
    labels = regime_series(frame.actual_cost_overrun_percentage)
    experts: dict[str, object | None] = {}
    strategies = {}
    for offset, regime in enumerate(REGIMES):
        subset = frame.loc[labels.eq(regime)].copy()
        if len(subset) < MIN_EXPERT_SAMPLES:
            experts[regime] = None
            strategies[regime] = {
                "samples": int(len(subset)),
                "strategy": "global_fallback_clipped",
                "minimum_required": MIN_EXPERT_SAMPLES,
            }
            continue
        model = _algorithm_regressor("catboost_mae_d4", SEED + 100 + offset)
        _fit_regressor(model, subset[FEATURES], subset.actual_cost_overrun_percentage, delay_target=False)
        experts[regime] = model
        strategies[regime] = {
            "samples": int(len(subset)),
            "strategy": "regime_specific_catboost",
            "minimum_required": MIN_EXPERT_SAMPLES,
        }
    return experts, strategies


def _clip_to_regime(predictions: np.ndarray, regime: str) -> np.ndarray:
    lower, upper = REGIME_BOUNDS[regime]
    return np.clip(np.asarray(predictions, dtype=float), lower, upper)


def expert_prediction_matrix(experts: dict, global_model, X: pd.DataFrame) -> np.ndarray:
    """Predict every expert without access to actual outcomes or actual regimes."""
    global_prediction = _predict_regressor(global_model, X, delay_target=False)
    columns = []
    for regime in REGIMES:
        expert = experts.get(regime)
        raw = global_prediction if expert is None else _predict_regressor(expert, X, delay_target=False)
        columns.append(_clip_to_regime(raw, regime))
    return np.column_stack(columns)


def classifier_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / math.log(len(REGIMES))


def confidence_alpha(probabilities: np.ndarray) -> np.ndarray:
    """Pre-declared confidence blend: chance-level confidence -> global fallback."""
    max_probability = probabilities.max(axis=1)
    chance = 1.0 / len(REGIMES)
    return np.clip((max_probability - chance) / (1.0 - chance), 0.0, 1.0)


def hard_route(probabilities: np.ndarray, expert_predictions: np.ndarray, global_predictions: np.ndarray) -> np.ndarray:
    selected = probabilities.argmax(axis=1)
    routed = expert_predictions[np.arange(len(expert_predictions)), selected]
    alpha = confidence_alpha(probabilities)
    return alpha * routed + (1.0 - alpha) * np.asarray(global_predictions, dtype=float)


def soft_route(probabilities: np.ndarray, expert_predictions: np.ndarray, global_predictions: np.ndarray) -> np.ndarray:
    mixture = np.sum(probabilities * expert_predictions, axis=1)
    alpha = confidence_alpha(probabilities)
    return alpha * mixture + (1.0 - alpha) * np.asarray(global_predictions, dtype=float)


def _metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 3),
        "median_absolute_error": round(float(median_absolute_error(actual, predicted)), 3),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mape": round(float(_safe_mape(actual, predicted)), 3),
    }


def _regime_mae(actual: pd.Series, predicted: np.ndarray) -> dict:
    labels = regime_series(actual).reset_index(drop=True)
    actual_values = actual.reset_index(drop=True).astype(float)
    output = {}
    for regime in REGIMES:
        mask = labels.eq(regime).to_numpy()
        output[regime] = {
            "projects": int(mask.sum()),
            "mae": round(float(mean_absolute_error(actual_values.to_numpy()[mask], predicted[mask])), 3) if mask.any() else None,
        }
    return output


def _error_distribution(actual, predicted) -> dict:
    errors = np.abs(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float))
    return {
        "within_5_pp": int((errors <= 5).sum()),
        "within_10_pp": int((errors <= 10).sum()),
        "within_20_pp": int((errors <= 20).sum()),
        "within_30_pp": int((errors <= 30).sum()),
        "over_30_pp": int((errors > 30).sum()),
        "p50_absolute_error": round(float(np.percentile(errors, 50)), 3),
        "p75_absolute_error": round(float(np.percentile(errors, 75)), 3),
        "p90_absolute_error": round(float(np.percentile(errors, 90)), 3),
        "p95_absolute_error": round(float(np.percentile(errors, 95)), 3),
        "maximum_error": round(float(errors.max()), 3),
    }


def _classification_report(actual_labels: pd.Series, probabilities: np.ndarray) -> dict:
    predicted_labels = np.asarray(REGIMES, dtype=object)[probabilities.argmax(axis=1)]
    actual = actual_labels.astype(str).to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted_labels, labels=REGIMES, zero_division=0)
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(actual), 1)), actual)
    majority_pred = majority.predict(np.zeros((len(actual), 1)))
    return {
        "accuracy": round(float(accuracy_score(actual, predicted_labels)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted_labels)), 4),
        "macro_precision": round(float(precision_score(actual, predicted_labels, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(actual, predicted_labels, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(actual, predicted_labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(actual, predicted_labels, average="weighted", zero_division=0)), 4),
        "majority_baseline_macro_f1": round(float(f1_score(actual, majority_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(actual, predicted_labels, labels=REGIMES).tolist(),
        "classes": REGIMES,
        "per_class": {
            regime: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, regime in enumerate(REGIMES)
        },
        "mean_max_probability": round(float(probabilities.max(axis=1).mean()), 4),
        "mean_normalized_entropy": round(float(classifier_entropy(probabilities).mean()), 4),
    }


def _internal_classifier_validation(train_data: pd.DataFrame) -> dict:
    validation_year = int(train_data.completion_year.max())
    fitting = train_data[train_data.completion_year < validation_year].copy()
    validation = train_data[train_data.completion_year == validation_year].copy()
    if len(fitting) < 12 or validation.empty:
        return {"status": "unavailable", "validation_year": validation_year}
    validation = apply_historical_priors(validation, historical_prior_maps(fitting))
    fitting = add_leave_one_out_training_priors(fitting)
    model = _fit_classifier(fitting, SEED + 50)
    probabilities = _ordered_probabilities(model, validation[FEATURES])
    report = _classification_report(regime_series(validation.actual_cost_overrun_percentage), probabilities)
    report.update({
        "status": "temporal_safe",
        "validation_year": validation_year,
        "training_projects": int(len(fitting)),
        "validation_projects": int(len(validation)),
        "policy": "Classifier architecture is checked on the latest training-window year; final holdout is not used for design choices.",
    })
    return report


def _decision(baseline_mae: float, experiment_mae: float) -> tuple[str, float, float]:
    absolute = baseline_mae - experiment_mae
    percentage = absolute / baseline_mae * 100.0
    if experiment_mae > baseline_mae:
        category = "WORSE"
    elif percentage >= 10.0:
        category = "STRONG SUCCESS"
    elif percentage >= 5.0:
        category = "POTENTIALLY USEFUL"
    else:
        category = "NEGLIGIBLE"
    return category, absolute, percentage


def run_experiment() -> dict:
    baseline_dir = MODELS / BASELINE_KEY
    metadata_path = baseline_dir / "metadata.json"
    evaluation_path = baseline_dir / "evaluation_results.json"
    cost_path = baseline_dir / "cost_model.pkl"
    delay_path = baseline_dir / "delay_model.pkl"
    risk_path = baseline_dir / "risk_model.pkl"
    registry_path = MODELS / "time_window_registry.json"
    before_hashes = {path.name: _sha256(path) for path in (cost_path, delay_path, risk_path, registry_path)}

    baseline_metadata = json.loads(metadata_path.read_text())
    baseline_metrics = json.loads(evaluation_path.read_text())
    if baseline_metadata["features_used"] != FEATURES:
        raise ValueError("Experiment 2 feature contract differs from production baseline.")

    window = window_for(BASELINE_KEY, baseline_metadata)
    all_data = labelled(outcome_data())
    train_data = all_data[all_data.completion_year.between(window.training_start, window.training_end)].copy()
    test_end = min(window.test_end, int(all_data.completion_year.max()))
    test_data = all_data[all_data.completion_year.between(window.test_start, test_end)].copy()
    if len(train_data) != int(baseline_metadata["training_samples"]):
        raise ValueError(f"Training cohort changed: {len(train_data)} vs baseline {baseline_metadata['training_samples']}")
    if len(test_data) != int(baseline_metadata["testing_samples"]):
        raise ValueError(f"Testing cohort changed: {len(test_data)} vs baseline {baseline_metadata['testing_samples']}")

    distribution = _regime_distribution(regime_series(train_data.actual_cost_overrun_percentage))
    internal_classifier = _internal_classifier_validation(train_data)

    model_train = add_leave_one_out_training_priors(train_data)
    test_model = apply_historical_priors(test_data, historical_prior_maps(train_data))

    classifier = _fit_classifier(model_train)
    global_model = _fit_global(model_train)
    experts, expert_strategies = _fit_experts(model_train)
    probabilities = _ordered_probabilities(classifier, test_model[FEATURES])
    expert_matrix = expert_prediction_matrix(experts, global_model, test_model[FEATURES])
    global_predictions = _predict_regressor(global_model, test_model[FEATURES], delay_target=False)
    hard_predictions = hard_route(probabilities, expert_matrix, global_predictions)
    soft_predictions = soft_route(probabilities, expert_matrix, global_predictions)

    actual = test_model.actual_cost_overrun_percentage.reset_index(drop=True)
    hard_metrics = _metrics(actual, hard_predictions)
    soft_metrics = _metrics(actual, soft_predictions)
    hard_metrics["regime_mae"] = _regime_mae(actual, hard_predictions)
    soft_metrics["regime_mae"] = _regime_mae(actual, soft_predictions)

    baseline_model = joblib.load(cost_path)
    baseline_predictions = _predict_regressor(baseline_model, test_model[FEATURES], delay_target=False)
    correction_artifact = {"cost": baseline_metadata.get("sector_correction_experiment", {}).get("cost", {})}
    baseline_predictions = apply_sector_correction(baseline_predictions, test_model, correction_artifact, "cost")
    computed_baseline = _metrics(actual, baseline_predictions)
    recorded_baseline_mae = float(baseline_metrics["cost_model"]["MAE"])
    if abs(computed_baseline["mae"] - recorded_baseline_mae) > 0.01:
        raise ValueError(f"Baseline reproduction failed: computed {computed_baseline['mae']} vs recorded {recorded_baseline_mae}")

    classifier_holdout = _classification_report(regime_series(actual), probabilities)
    classifier_holdout["evaluation_only"] = True
    classifier_holdout["note"] = "Final holdout classifier metrics are reported after architecture was fixed; they were not used for threshold/model selection."

    hard_decision, hard_abs, hard_pct = _decision(recorded_baseline_mae, hard_metrics["mae"])
    soft_decision, soft_abs, soft_pct = _decision(recorded_baseline_mae, soft_metrics["mae"])
    if soft_metrics["mae"] <= hard_metrics["mae"]:
        best_name, best_metrics, decision, absolute, percentage = "soft_mixture", soft_metrics, soft_decision, soft_abs, soft_pct
        best_predictions = soft_predictions
    else:
        best_name, best_metrics, decision, absolute, percentage = "hard_routing", hard_metrics, hard_decision, hard_abs, hard_pct
        best_predictions = hard_predictions

    error_distribution = {
        "baseline": _error_distribution(actual, baseline_predictions),
        "hard_routing": _error_distribution(actual, hard_predictions),
        "soft_mixture": _error_distribution(actual, soft_predictions),
    }

    errors = np.abs(best_predictions - actual.to_numpy(dtype=float))
    largest = np.argsort(errors)[-20:][::-1]
    probability_rows = []
    project_names = test_model.get("project_name", pd.Series([""] * len(test_model), index=test_model.index)).reset_index(drop=True)
    project_ids = test_model.get("project_id", pd.Series([""] * len(test_model), index=test_model.index)).astype(str).reset_index(drop=True)
    for index in largest:
        probability_rows.append({
            "project_id": project_ids.iloc[index],
            "project_name": str(project_names.iloc[index]),
            "actual_overrun": round(float(actual.iloc[index]), 4),
            "baseline_prediction": round(float(baseline_predictions[index]), 4),
            "hybrid_prediction": round(float(best_predictions[index]), 4),
            "absolute_baseline_error": round(float(abs(baseline_predictions[index] - actual.iloc[index])), 4),
            "absolute_hybrid_error": round(float(errors[index]), 4),
            "predicted_regime_probabilities": {regime: round(float(probabilities[index, j]), 6) for j, regime in enumerate(REGIMES)},
            "classifier_selected_regime": REGIMES[int(probabilities[index].argmax())],
        })

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, EXPERIMENT_DIR / "regime_classifier.pkl")
    joblib.dump(global_model, EXPERIMENT_DIR / "global_fallback_cost_model.pkl")
    for regime, expert in experts.items():
        if expert is not None:
            joblib.dump(expert, EXPERIMENT_DIR / f"{regime.lower()}_regressor.pkl")

    prediction_frame = pd.DataFrame({
        "project_id": project_ids,
        "project_name": project_names,
        "actual_cost_overrun": actual,
        "baseline_prediction": baseline_predictions,
        "hard_routing_prediction": hard_predictions,
        "soft_mixture_prediction": soft_predictions,
        "actual_regime_evaluation_only": regime_series(actual).to_numpy(),
        "predicted_regime": np.asarray(REGIMES, dtype=object)[probabilities.argmax(axis=1)],
        "classifier_max_probability": probabilities.max(axis=1),
        "classifier_normalized_entropy": classifier_entropy(probabilities),
    })
    for index, regime in enumerate(REGIMES):
        prediction_frame[f"probability_{regime.lower()}"] = probabilities[:, index]
    prediction_frame.to_csv(EXPERIMENT_DIR / "prediction_validation.csv", index=False)

    result = {
        "experiment": EXPERIMENT_NAME,
        "baseline": {
            "model_version": BASELINE_KEY,
            "cost_mae": recorded_baseline_mae,
            "computed_reproduction": computed_baseline,
            "delay_mae": float(baseline_metrics["delay_model"]["MAE_days"]),
            "feature_count": len(FEATURES),
        },
        "cohort": {
            "training_years": [window.training_start, window.training_end],
            "testing_years": [window.test_start, test_end],
            "training_samples": int(len(train_data)),
            "testing_samples": int(len(test_data)),
            "features": FEATURES,
            "seed": SEED,
        },
        "regime_distribution": distribution,
        "internal_classifier_validation": internal_classifier,
        "classifier": classifier_holdout,
        "expert_strategies": expert_strategies,
        "hard_routing": hard_metrics,
        "soft_mixture": soft_metrics,
        "best_experiment_variant": best_name,
        "best_experiment_mae": best_metrics["mae"],
        "absolute_improvement": round(float(absolute), 3),
        "percentage_improvement": round(float(percentage), 2),
        "decision": decision,
        "routing_policy": {
            "hard": "argmax regime expert blended with global fallback using pre-declared classifier confidence alpha",
            "soft": "probability-weighted expert mixture blended with global fallback using pre-declared classifier confidence alpha",
            "alpha": "clip((max_class_probability - 0.2) / 0.8, 0, 1); never tuned on final holdout",
            "expert_bounds": REGIME_BOUNDS,
            "oracle_routing": False,
        },
        "production_safety": {
            "production_models_modified": False,
            "registry_modified": False,
            "project_history_used": False,
            "monthly_extraction_added": False,
        },
    }

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "baseline_model_version": BASELINE_KEY,
        "features_used": FEATURES,
        "regime_thresholds": {
            "COST_SAVING": "<=0",
            "LOW": "(0,20]",
            "MEDIUM": "(20,100]",
            "HIGH": "(100,200]",
            "EXTREME": ">200",
        },
        "training_years": [window.training_start, window.training_end],
        "testing_years": [window.test_start, test_end],
        "training_samples": int(len(train_data)),
        "testing_samples": int(len(test_data)),
        "classifier_model_type": "CatBoostClassifier MultiClass auto_class_weights=Balanced",
        "expert_model_type": "catboost_mae_d4",
        "expert_strategies": expert_strategies,
        "fallback_strategy": "global baseline-style CatBoost; low-confidence routing and undersized regimes fall back toward it",
        "routing_strategies": ["hard_routing", "soft_mixture"],
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(metadata, EXPERIMENT_DIR / "metadata.json")
    _write_json(distribution, EXPERIMENT_REPORTS / "exp2_regime_distribution.json")
    _write_json({"internal_temporal": internal_classifier, "final_holdout_evaluation_only": classifier_holdout}, EXPERIMENT_REPORTS / "exp2_regime_classifier_metrics.json")
    _write_json(error_distribution, EXPERIMENT_REPORTS / "exp2_error_distribution.json")
    _write_json({"best_variant": best_name, "largest_errors": probability_rows}, EXPERIMENT_REPORTS / "exp2_extreme_case_analysis.json")
    _write_json(result, EXPERIMENT_REPORTS / "v3_hybrid_cost_regime.json")

    evolution_path = EXPERIMENT_REPORTS / "model_evolution.json"
    evolution = json.loads(evolution_path.read_text()) if evolution_path.exists() else []
    evolution = [row for row in evolution if row.get("version") != "v3"]
    evolution.append({
        "version": "v3",
        "experiment": EXPERIMENT_NAME,
        "cost_mae": best_metrics["mae"],
        "improvement_percentage": round(float(percentage), 2),
        "decision": decision,
    })
    _write_json(evolution, evolution_path)

    after_hashes = {path.name: _sha256(path) for path in (cost_path, delay_path, risk_path, registry_path)}
    if before_hashes != after_hashes:
        raise RuntimeError("Experiment 2 modified a production artifact or registry.")

    print("EXPERIMENT2_RESULT_JSON=" + json.dumps({
        "baseline_mae": recorded_baseline_mae,
        "baseline_rmse": computed_baseline["rmse"],
        "baseline_r2": computed_baseline["r2"],
        "hard_mae": hard_metrics["mae"],
        "hard_rmse": hard_metrics["rmse"],
        "hard_r2": hard_metrics["r2"],
        "soft_mae": soft_metrics["mae"],
        "soft_rmse": soft_metrics["rmse"],
        "soft_r2": soft_metrics["r2"],
        "best_variant": best_name,
        "best_mae": best_metrics["mae"],
        "absolute_improvement": round(float(absolute), 3),
        "percentage_improvement": round(float(percentage), 2),
        "classifier_macro_f1": classifier_holdout["macro_f1"],
        "classifier_accuracy": classifier_holdout["accuracy"],
        "classifier_majority_macro_f1": classifier_holdout["majority_baseline_macro_f1"],
        "decision": decision,
        "training_samples": int(len(train_data)),
        "testing_samples": int(len(test_data)),
        "production_hashes_unchanged": before_hashes == after_hashes,
    }, sort_keys=True))
    return result


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
