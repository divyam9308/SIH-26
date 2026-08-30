# Experiment 62 — U1 nonlinear OOF residual booster

Baseline assumption: PR #96 / Experiment 61 is production.

This challenger keeps the Exp61 Cost and Delay predictions as the anchor and learns separate, heavily regularized LightGBM residual corrections from rolling out-of-fold training errors only. Corrections are capped by the training-only weighted 90th percentile absolute residual. The future holdout is never used for fitting, feature selection, correction caps, or routing.

Evaluation runs both 2001–2019 and 2001–2021 through 2025 on the exact same production/challenger cohort. The 2001–2021 run must retain 721 projects / 11,200 snapshots. Scientific regression remains a green execution result and never auto-promotes or writes production artifacts.
