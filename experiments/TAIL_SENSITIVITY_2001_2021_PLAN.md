# Tail Sensitivity Diagnostic — 2001–2021 production window

## Purpose

Measure how strongly highly skewed project outcomes affect Cost and Delay MAE, RMSE, and R² on the frozen 2022–2025 holdout without changing the production model, training data, holdout cohort, or headline official evaluation.

## Leakage-safe threshold policy

Tail cutoffs are derived only from unique-project actual outcomes in the 2001–2021 training population. The experiment records P90, P95, and P99 thresholds independently for:

- actual cost overrun percentage
- actual delay days

No 2022–2025 prediction error is used to choose thresholds or exclusions.

## Diagnostic cohorts

For each target, report metrics for:

- all holdout projects
- normal <= P90
- P90–P95
- P95–P99
- > P99
- excluding top 5% (<= P95)
- excluding top 1% (<= P99)
- tail > P95

The output must include project counts and rows/snapshots for every cohort.

## Metrics

- MAE
- RMSE
- R²
- delta versus the full frozen holdout for the <=P95 and <=P99 diagnostic subsets

## Interpretation

The full holdout remains the official primary evaluation. Trimmed-cohort metrics are diagnostic only and must not be used to claim improved production accuracy or to promote a model.

A large positive R² delta and large negative MAE/RMSE deltas after excluding the training-defined tail indicate that extreme-outcome projects materially dominate headline error. That result should motivate tail-aware modelling or routing, not deletion of difficult projects from production evaluation.
