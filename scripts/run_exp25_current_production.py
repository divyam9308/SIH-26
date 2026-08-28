"""Run Exp25 against the exact current production Cost/Delay stack.

This fast scientific harness reproduces Exp12 Cost and Exp34 Delay training,
while skipping risk fitting, SHAP, artifact publication and other production-only
work that cannot affect Cost/Delay MAE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.path_oof_delay_exp34 import (
    PATH_FEATURES,
    _fit_delay_family_models,
    _oof_delay_weights,
    enrich_path_dependence,
)
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production
from backend.app.ml.production_delay_baseline import (
    PRODUCTION_DELAY_BASELINE,
    ProductionDelayBlendModel,
)
from scripts.run_fast_current_experiment import fast_current_production


def exact_current_cost_delay_production(
    data: pd.DataFrame, start: int, end: int, test_end: int
) -> tuple[dict, dict]:
    bundle, receipt = fast_current_production(data, start, end, test_end)

    enriched = enrich_path_dependence(enrich_supervised_for_production(data.copy()))
    enriched["completion_year"] = pd.to_numeric(
        enriched["completion_year"], errors="coerce"
    )
    enriched["snapshot_date"] = pd.to_datetime(
        enriched["snapshot_date"], errors="coerce"
    )
    train, _ = temporal_project_split(enriched, start, end, test_end)

    base_delay_features = list(bundle["metadata"].get("delay_features_used") or [])
    delay_features = list(dict.fromkeys(base_delay_features + PATH_FEATURES))
    delay_weights, delay_oof = _oof_delay_weights(train, delay_features)
    delay_models = _fit_delay_family_models(train, delay_features)
    bundle["delay"] = ProductionDelayBlendModel(
        delay_models, delay_weights, delay_features
    )

    metadata = bundle["metadata"]
    metadata["delay_features_used"] = delay_features
    metadata["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    metadata["promoted_delay_from_experiment"] = "exp_34"
    metadata["delay_blend_weights"] = delay_weights
    metadata["delay_rolling_oof"] = delay_oof
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["delay"] = "exp34_oof_blend"
    metadata["selected_algorithms"] = selected

    receipt = dict(receipt)
    receipt["selected_algorithms"] = selected
    receipt["production_delay_baseline"] = PRODUCTION_DELAY_BASELINE
    receipt["delay_blend_weights"] = delay_weights
    receipt["delay_rolling_oof"] = delay_oof
    return bundle, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--test-end", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")

    bundle, receipt = exact_current_cost_delay_production(
        data, args.start, args.end, args.test_end
    )
    adapter = get_experiment_adapter("exp_25_current")
    fitted = adapter.module.fit_against_production(
        data=data,
        training_start=args.start,
        training_end=args.end,
        test_end=args.test_end,
        production_bundle=bundle,
        production_receipt=receipt,
    )

    payload = {
        "window": f"{args.start}_{args.end}",
        "test_end": args.test_end,
        "audit_mode": "exact current Exp12 Cost + Exp34 Delay; risk/SHAP/publication skipped",
        "production": receipt,
        "production_metadata": bundle["metadata"],
        "experiment": dict(fitted.get("experiment") or {}),
        "overall_comparison": dict(fitted.get("overall_comparison") or {}),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n"
    )
    print(
        "CURRENT_PRODUCTION_COMPARISON="
        + json.dumps(
            payload["overall_comparison"],
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
