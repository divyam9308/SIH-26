# Experiment 18 — GRU with temporal attention

## Hypothesis
The rejected plain GRU compressed each project history into only its final recurrent hidden state. This challenger keeps the recurrent encoder but learns an attention distribution over all valid past monthly states, allowing old cost revisions, deterioration episodes or recoveries to receive explicit weight.

## Controlled contract
- Fresh branch from current `main`; production remains unchanged.
- Production cost baseline remains `exp12_trajectory_v3_cost_only`.
- Every prediction at `t` sees only reports with `snapshot_date <= t`.
- Same cohort, temporal split, project-balanced weights and future holdout as the other neural challengers.
- 12/24/36/60/full history are selected only on up to three forward temporal folds inside training.
- Cost and delay choose history length independently.
- No automatic promotion.

## Architecture
One-layer 48-unit GRU, learned additive temporal attention over all non-padding hidden states, categorical embeddings, static numeric context, fusion head, standardized cost/delay targets and project-balanced Smooth-L1 objective.

## Required audit
Run 2001–2019 and 2001–2021 controlled comparisons. Report production and challenger MAE, percent improvement, selected history length, all five internal history-ablation summaries, lifecycle-stage metrics, stage-balanced metrics and paired-project bootstrap evidence.
