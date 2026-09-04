# Delay R² experiment — 2001–2021

Objective: improve delay R² on the current canonical lifecycle model without changing the frozen 2022–2025 holdout.

Method:
- Preserve current AFT + residual calibration + fallback routing as baseline.
- Diagnose rolling-OOF delay residuals by sector, lifecycle stage, delay tail, and AFT-vs-fallback routing.
- Focus candidate correction on the fallback group and large residual tails using training-only rolling OOF data.
- Keep all feature inputs strictly as-of and pre-completion.
- Select correction/calibration parameters on historical OOF folds only.
- Evaluate once on frozen 2022–2025.

Promotion gate:
- Delay R² must improve.
- Delay MAE must not materially regress.
- Existing AFT routing and frozen reference semantics remain intact unless the experiment explicitly demonstrates a leakage-safe improvement.
- No tuning on 2022–2025 outcomes.
