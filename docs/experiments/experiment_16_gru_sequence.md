# Experiment 16 — GRU monthly-sequence forecasting

## Status

**EXPERIMENTAL / NOT PRODUCTION**

Experiment 16 is a fresh challenger created directly from `main`, where the current production cost path is the promoted Experiment 12 trajectory model and delay remains the existing production lifecycle model.

This experiment does not promote, overwrite, or retrain production artifacts.

## Hypothesis

Experiment 12 proved that leakage-safe monthly history contains useful forecasting signal, while Experiment 13 showed that repeatedly hand-combining those summary features does not reliably improve cost MAE. Experiment 16 therefore changes the representation itself:

> Can a compact GRU learn predictive structure directly from the ordered monthly PAIMANA report sequence that is lost when history is reduced to fixed 3/6/12-month summary features?

## Input sequence

For a supervised prediction snapshot at date `t`, the GRU receives only official monthly reports for the same canonical project satisfying:

`monthly_report.snapshot_date <= t`

No report after the prediction date is available to the model.

The sequence currently uses numeric monthly signals available from the trajectory dataset, including approved/revised cost, cumulative expenditure, physical progress, schedule slippage, planned duration and expected progress where present. Missingness is supplied explicitly as an additional mask channel.

The final sequence embedding is combined with leakage-safe current static context:

- sector;
- implementing agency;
- project-size category;
- approved cost;
- planned duration;
- elapsed duration;
- duration ratio.

Categorical vocabularies and all numeric scalers are fit on training projects only.

## History-length ablation

This is **one experiment PR**, not five independent future-holdout experiments.

Five sequence windows are candidates:

1. 12 months
2. 24 months
3. 36 months
4. 60 months
5. full available as-of history

History length is selected strictly inside the training period using up to three rolling completion-year validation folds. Cost and delay select their history length independently.

Example valid outcome:

- cost selects full history;
- delay selects 36 months.

The future holdout is not consulted when making this choice. This prevents repeatedly trying different windows against the final test cohort and selecting the one that happened to look best.

## Neural architecture

The challenger is intentionally compact for the available project count and CPU-controlled audit environment:

- one-layer GRU over the ordered monthly sequence;
- 48-dimensional recurrent hidden state;
- learned embeddings for sector, agency and project-size category;
- small static-context MLP;
- shared fusion layer;
- two regression outputs: final cost-overrun percentage and final delay days;
- weighted Smooth-L1 training loss on standardized targets;
- project-balanced snapshot weights inherited from the supervised dataset;
- AdamW optimizer and gradient clipping.

The network is multi-task during training, but cost and delay may use models trained with different selected history lengths after internal validation.

## Sampling policy

The prediction/evaluation rows remain the existing project-balanced supervised snapshots. A quarterly prediction row may consume **all preceding monthly reports**. This keeps every monthly report available to the sequence encoder without allowing very long projects to dominate merely because they contain more monthly records.

A sequence needs at least three available monthly reports.

## Temporal selection

For each training window, up to the final three valid completion years are used as rolling forward validation years. For each fold:

1. preprocessing is fit only on projects completing before the validation year;
2. all five history lengths are trained with the same architecture/training policy;
3. project-balanced cost and delay MAE are recorded on the validation year;
4. mean validation MAE across folds selects history length separately for cost and delay.

Only after selection is complete are final models refit on the full training period and evaluated on the untouched future cohort.

## Required controlled windows

The audit workflow runs both standard comparisons:

- train 2001–2019, test on later completed projects;
- train 2001–2021, test on later completed projects.

For each window report:

- production cost MAE;
- GRU cost MAE;
- cost MAE improvement percentage;
- production delay MAE;
- GRU delay MAE;
- delay MAE improvement percentage;
- selected history length for each target;
- paired-project bootstrap probability and 95% improvement interval;
- lifecycle-stage MAE;
- stage-balanced MAE;
- comparable project/snapshot counts;
- complete internal history-length ablation results.

## Scientific interpretation

A green GitHub Actions run proves only that the implementation executed. It does **not** mean the neural model is better.

Positive improvement means lower MAE than current production. Cost and delay are judged independently. A gain in one target must not be used to justify promotion of a regressed target.

## Promotion rule

No automatic promotion is allowed.

If the neural challenger produces reproducible improvement across the required windows with acceptable paired-project evidence and lifecycle behavior, promotion should occur in a separate deliberate production PR containing only the winning target/configuration.
