# Experiment 5 verification against current production

This verification branch ports the Experiment 5 common-holdout implementation from `adityaab2007/Krish-SIH26` onto the current `divyam9308/SIH-26` production baseline.

## Required comparison

For each training window below, both models are evaluated on the exact same fixed 2022–2025 cohort:

- 2001–2019 training → 2022–2025 holdout
- 2001–2021 training → 2022–2025 holdout

The production side uses the currently promoted Experiment 12 trajectory-enhanced cost baseline. Delay and risk retain the existing production contract. The Experiment 5 side preserves Krish's audited 25-feature lifecycle implementation and original seeds.

The dedicated workflow writes `reports/experiments/exp5_vs_current_production.{json,md}` and reports cost/delay MAE, RMSE, R², lifecycle-stage diagnostics, feature contracts, algorithms, cohort size/fingerprint and paired-project evidence.

This PR is verification-only and should not be merged as-is.
