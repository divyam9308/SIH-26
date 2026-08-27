# Experiment 17 — Causal TCN monthly sequence model

## Hypothesis
A causal dilated Temporal Convolutional Network may learn repeated deterioration, recovery, revision cascades and long-range trajectory motifs from ordered PAIMANA monthly reports better than recurrent compression.

## Controlled contract
- Isolated challenger; production remains unchanged.
- Current production cost baseline remains `exp12_trajectory_v3_cost_only`.
- Input at prediction snapshot `t` contains only reports for that project with `snapshot_date <= t`.
- Same identity-verified supervised cohort, temporal project split and project-balanced weighting as production comparison.
- Cost and delay are evaluated independently.
- No automatic promotion.

## History-length ablation
One PR evaluates five internal candidates: 12, 24, 36, 60 months and full available history. Up to three forward-only completion-year folds inside the training period choose history length separately for cost and delay. The future holdout is not inspected during selection.

## Architecture
- causal 1-D convolutional input projection
- residual dilated blocks with dilations 1, 2, 4, 8, 16, 32
- kernel size 3, giving a long receptive field without recurrent hidden-state compression
- learned sector/agency/project-size embeddings
- static numeric context fusion
- two-output cost/delay head
- project-balanced Smooth-L1 training objective

## Required audit
Run the controlled comparison for both 2001–2019 and 2001–2021 training windows. Report production vs TCN cost MAE, delay MAE, percentage improvement, selected history length, lifecycle-stage diagnostics, stage-balanced MAE and paired-project bootstrap evidence.

A green workflow establishes technical correctness only. Scientific acceptance requires reproducible improvement on the untouched future cohorts.
