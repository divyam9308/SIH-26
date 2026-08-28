# Exp33-on-Exp34 Delay ablation

This PR isolates the Experiment 33 weighted-median residual calibration on top of the current production Delay path promoted from Experiment 34.

## What changes

- Current Exp34 Delay features, model families, and OOF-selected production blend remain the baseline.
- Exp33 learns a post-model weighted-median residual correction from rolling historical validation predictions.
- Corrections use lifecycle stage plus prediction bin, with bin/global fallbacks.

## What does not change

- Production `main` is untouched by the experiment.
- Cost remains the exact promoted Exp12 production model and is not calibrated.
- No production artifacts are overwritten.
- The 2022-2025 future holdout is never used to fit calibration.

## Evaluation contract

Actions evaluates both:

- 2001-2019 -> future through 2025
- 2001-2021 -> 2022-2025

For 2001-2021, CI requires the exact verified shared production cohort and baseline before accepting the comparison:

- 721 projects
- 11,200 snapshots
- production Cost MAE 26.872
- production Exp34 Delay MAE 501.303

A regression is a valid scientific result and must not fail CI solely because the challenger is worse. Promotion is a separate explicit decision.
