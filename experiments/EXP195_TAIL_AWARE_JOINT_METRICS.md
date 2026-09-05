# Experiment 195 — Tail-aware joint Cost/Delay metric optimization

## Hypothesis
PR #194 showed that the hardest Cost and Delay tails materially inflate MAE/RMSE, while removing them also destroys useful target variance and reduces R². Exp195 therefore keeps every frozen holdout project and tests a tail-aware residual correction layer instead of trimming projects.

## Frozen comparison contract
- Training: 2001–2021 only.
- Evaluation: full 2022–2025 frozen holdout.
- No 2022–2025 outcome is used for feature choice, tail thresholds, residual-model fitting, scale selection, or hyperparameter selection.
- Current Exp105 Cost + Exp113 Delay production remains the anchor.
- Risk is untouched.
- Production artifacts are not promoted or overwritten.

## Candidate change
For Cost and Delay independently:
1. Build forward temporal OOF residual evidence inside 2001–2021.
2. Derive P90/P95 target thresholds only from that training OOF evidence.
3. Increase training weight for difficult tail observations instead of deleting them.
4. Fit a regularized shallow LightGBM residual corrector using only as-of execution features plus the production prediction.
5. Select correction scale using forward meta-OOF evidence with a joint objective covering MAE, RMSE, and R².
6. Apply the selected correction on top of the unchanged production anchor to the complete frozen holdout.

## Acceptance rule
Exp195 is ACCEPT only if, on the full frozen 2022–2025 holdout:
- Cost MAE decreases,
- Cost RMSE decreases,
- Cost R² increases,
- Delay MAE decreases,
- Delay RMSE decreases, and
- Delay R² increases.

Any trade-off fails the experiment. A successful workflow with worse scientific metrics remains a valid REJECT and must not be tuned against the holdout.
