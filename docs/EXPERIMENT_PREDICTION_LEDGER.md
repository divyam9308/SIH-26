# Experiment prediction ledger contract

Every new scientific experiment should persist the exact row-level evidence used for its headline MAE comparison. The purpose is to make later slice analysis, combination analysis, and experiment post-mortems reproducible without retraining or relying on copied PR summary numbers.

This is evidence infrastructure only. It must not change production prediction behavior or production artifact locations.

## Canonical artifacts

Each experiment run should write these files into its immutable experiment run directory:

- `prediction_ledger.csv`
- `prediction_ledger_manifest.json`

`backend.app.ml.experiments.prediction_ledger.write_experiment_prediction_ledger` resolves the standard experiment-only path under `models/monthly_lifecycle/experiments/**`. The existing generic experiment workflow already uploads that tree.

## Required row identity and weighting

Every ledger row must include:

- `canonical_project_id`
- `snapshot_date`
- `sample_weight`
- one `experiment_id`
- one training `window`

The `(canonical_project_id, snapshot_date)` key must be unique. After all experiment-specific filtering, `sample_weight` must be recomputed so every project sums to exactly 1 (within numerical tolerance). This prevents repeated snapshots from giving long-history projects more influence.

The manifest fingerprints the exact project/snapshot/weight cohort separately from the prediction values. A challenger cannot claim a paired comparison if its ledger cohort fingerprint differs from the rows used for baseline scoring.

## Target columns

A Cost ledger contains:

- `actual_cost_overrun_percentage`
- `production_cost_prediction`
- `experiment_cost_prediction`

A Delay ledger contains:

- `actual_delay_days`
- `production_delay_prediction`
- `experiment_delay_prediction`

A target-specific experiment may persist only its changed target. It must not invent evidence for an unchanged target. If a workflow intentionally reports an unchanged target, pass the actual production and challenger predictions explicitly.

The builder also records production/challenger absolute error and per-row absolute-error improvement. Positive row improvement always means the challenger reduced absolute error.

## Slice columns

When present in the scored dataframe, the standard builder carries useful diagnostic columns such as:

- completion year
- lifecycle stage
- sector
- implementing agency
- state
- scale bucket / approved cost
- current cost escalation
- schedule slippage
- duration ratio
- history depth
- parser family
- experiment route (for example specialist vs fallback)

Experiment-specific safe columns can be supplied with `extra_columns`. Do not add target-derived or future information merely to make a later slice easier.

## Minimal usage

```python
from backend.app.ml.experiments.prediction_ledger import (
    build_prediction_ledger,
    write_experiment_prediction_ledger,
)

ledger = build_prediction_ledger(
    comparable_rows,
    experiment_id="exp_45",
    window="2001_2021",
    production_cost_prediction=production_cost_pred,
    experiment_cost_prediction=experiment_cost_pred,
)

write_experiment_prediction_ledger(
    ledger,
    experiment_id="exp_45",
    window="2001_2021",
    run_id=run_id,
    extra_manifest={
        "primary_target": "cost",
        "execution_verdict": "EXECUTION VALID",
        "scientific_verdict": verdict,
    },
)
```

## Required workflow behavior for future experiments

For both 2001–2019 and 2001–2021 comparisons:

1. Freshly construct the exact production and challenger comparison cohort.
2. Require identical paired observation keys.
3. Recalculate project-balanced weights after all filtering.
4. Generate production and challenger predictions on those exact rows.
5. Build and validate the prediction ledger.
6. Assert the ledger cohort matches the scored dataframe.
7. Persist the immutable ledger and manifest.
8. Upload the experiment artifact directory even when the scientific result is a regression.
9. Keep execution validity separate from `PROMOTION CANDIDATE` / `DO NOT PROMOTE`.

The ledger is evidence, not a promotion mechanism. No experiment is allowed to overwrite production artifacts or promote itself.
