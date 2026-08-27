# Experiment 20 — LSTM monthly sequence model

## Hypothesis
A compact LSTM may preserve useful long-horizon project state more effectively than the rejected plain GRU because its explicit cell state and gating can retain or forget trajectory information over longer histories.

## Controlled contract
- Fresh isolated branch from current `main`; production remains unchanged.
- Production cost baseline remains `exp12_trajectory_v3_cost_only`.
- Every prediction at snapshot `t` receives only official reports with `snapshot_date <= t`.
- Same identity-verified cohort, temporal project split, project-balanced weights and final future holdout as the other neural challengers.
- 12/24/36/60/full history lengths are selected only on forward temporal validation folds inside training.
- Cost and delay select history independently; no automatic promotion.

## Architecture
One-layer 48-unit LSTM, categorical embeddings, static numeric context, fusion head, standardized cost/delay targets and project-balanced Smooth-L1 training objective.

## Required audit
Run 2001–2019 and 2001–2021 controlled comparisons. Report production/challenger cost MAE, delay MAE, percentage improvement, selected history lengths, all five internal history variants, lifecycle-stage metrics, stage-balanced MAE and paired-project bootstrap evidence.
