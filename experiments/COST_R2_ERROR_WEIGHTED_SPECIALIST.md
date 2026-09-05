# Cost R² Experiment — strict-OOF error-weighted squared-error specialist

## Objective
Raise Cost R² while preserving Cost MAE on the frozen 2001–2021 -> 2022–2025 comparison.

## Difference from the MSE-blend experiment
This PR keeps the specialist family fixed to a regularized LightGBM L2 model and changes the training distribution instead. Projects that the complete current production Cost stack predicted poorly in strict forward OOF receive larger training weight.

For every project represented in prior strict OOF evidence:

`project_error = mean(production_oof_residual^2)`

The error is normalized by the median positive project error and converted to a bounded multiplier. Candidate strengths are selected only from training OOF evidence, with the multiplier capped at 4x so a few extreme projects cannot dominate training.

## Nested leakage-safe selection
For each meta validation year:
1. Production errors from strictly earlier OOF years define project weights.
2. A weighted L2 specialist is fitted only on projects completed before the validation year.
3. That specialist predicts the validation year.

After all meta folds are collected, weighting strength and blend alpha are selected by lowest OOF RMSE subject to OOF MAE <= production OOF MAE.

The final weighted specialist is then refit on the full 2001–2021 training set using weights derived from all strict training OOF evidence. Only then is 2022–2025 evaluated.

## Acceptance
A scientific promotion candidate requires:
- non-zero selected error-weight strength,
- non-zero selected specialist blend,
- Cost MAE <= production,
- Cost RMSE < production,
- Cost R² > production.

The full holdout remains intact. Delay/Risk and production artifacts are untouched. Green CI is execution validity only.
