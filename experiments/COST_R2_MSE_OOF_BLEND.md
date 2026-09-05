# Cost R² Experiment — Independent MSE specialist + OOF blend

## Objective
Increase Cost R² without degrading Cost MAE on the frozen 2001–2021 -> 2022–2025 production comparison.

## Candidate
The current Exp105 Cost prediction stays the anchor. A completely independent Cost specialist is trained directly on the same leakage-safe Cost input contract using squared-error objectives. Training-only strict forward production OOF evidence selects both:

1. the squared-error family (LightGBM L2, XGBoost squared error, or ExtraTrees squared error), and
2. convex blend alpha in 0.05 increments from 0 through 1.

The selection objective is lowest OOF RMSE subject to blended OOF MAE being no worse than current production OOF MAE.

Final prediction:

`candidate = (1 - alpha) * current_production + alpha * mse_specialist`

## Leakage controls
- Every production OOF fold retrains the complete current production stack only through the year before that fold.
- Specialist fold fitting also uses only projects completed before the fold year.
- Family and alpha are frozen before evaluating 2022–2025.
- The full frozen holdout is retained; no tail rows/projects are removed.
- No production artifact is overwritten and no model is auto-promoted.

## Acceptance
Scientific verdict is `PROMOTION CANDIDATE` only when, on the unchanged frozen holdout:
- Cost MAE <= production Cost MAE,
- Cost RMSE < production Cost RMSE, and
- Cost R² > production Cost R².

A green workflow means execution was valid, not that the experiment won.
