# Experiment 35 — Exp32 + Exp33

This branch is an isolated challenger against current production.

Current production:
- Cost: Exp12 trajectory model.
- Delay: Exp34 path-dependence + rolling-OOF ExtraTrees/LightGBM/XGBoost blend.

Combined challenger:
- Cost: Exp33 cross-fitted weighted-median residual calibration. Exp32 is Delay-only and therefore cannot affect Cost directly.
- Delay: Exp32 AFT-style `log1p(remaining days)` target using the current Exp34 Delay feature contract and fixed current Exp34 blend weights, followed by Exp33 cross-fitted residual calibration.

Evaluation:
- Cost uses the shared Exp12-comparable production cohort. For 2001-2021 this must be exactly 721 projects / 11,200 snapshots and must reproduce production Cost MAE 26.872.
- The current Exp34 Delay baseline is also checked on that full shared cohort and must reproduce 501.303 days for 2001-2021.
- Fair AFT Delay comparison uses the subset of the shared cohort with planned-completion evidence and a positive retrospective remaining-time interval; production Delay is scored on exactly the same subset.
- Rolling calibration folds are entirely inside the training window. Future holdout outcomes are never used to fit calibration or select weights.

No production artifact is overwritten and no automatic promotion is allowed.
