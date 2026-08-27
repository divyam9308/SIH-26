# Experiment 19 — Small Transformer monthly sequence model

## Hypothesis
A compact self-attention encoder may capture long-range relationships between early revisions, later expenditure slowdown, progress changes and schedule deterioration that recurrent models compress poorly.

## Controlled contract
- Fresh isolated branch from current `main`; production is untouched.
- Production cost baseline remains `exp12_trajectory_v3_cost_only`.
- Input contains only official reports with `snapshot_date <= prediction snapshot`.
- Same identity-verified cohort, temporal project split, project-balanced weights and final future holdout as the other neural challengers.
- 12/24/36/60/full history variants are chosen only on forward-only validation folds inside the training period.
- Cost and delay select history independently; no automatic promotion.

## Architecture
Two-layer Transformer encoder, 64-dimensional token representation, four attention heads, 128-dimensional feed-forward blocks, sinusoidal positions, causal attention mask, padding mask, categorical embeddings and static context fusion. Training uses standardized cost/delay targets and project-balanced Smooth-L1 loss.

## Required audit
Run controlled 2001–2019 and 2001–2021 comparisons and report production/challenger cost MAE, delay MAE, percentage improvement, selected history lengths, all five internal history variants, lifecycle-stage metrics, stage-balanced MAE and paired-project bootstrap evidence.
