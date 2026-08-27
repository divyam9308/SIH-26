# Experiment 13 — Advanced trajectory/regime interaction model

## Status

**V2 EXPERIMENTAL / PENDING CONTROLLED EVIDENCE**

Experiment 13 v1 executed successfully but did not produce a reproducible improvement over production. It is retained in the branch for reproducibility, while the active adapter now points to the redesigned v2 implementation.

This PR remains experiment-only. It does **not** modify or promote production.

Current production remains:

- cost: promoted Experiment 12 trajectory-enhanced cost baseline;
- delay: existing production lifecycle delay model;
- risk: existing production lifecycle risk model.

## Why v1 was redesigned

V1 mostly recombined information already available to the promoted Experiment 12 representation. Its hand-designed pressure scores and explicit interactions added complexity but little genuinely new information, and the final future holdouts showed no stable gain.

The v1 controlled result was therefore treated as negative evidence rather than promoted.

V2 directly addresses the three observed failure modes:

1. learn trajectory regimes from the training data instead of prescribing them with hand-built pressure formulas;
2. require candidate features to generalize across multiple rolling historical validation folds rather than one final two-year block;
3. explicitly prioritize early/mid lifecycle forecasting during challenger fitting and model selection, while keeping the final holdout evaluation project-balanced.

## V2 hypothesis

> Latent project states learned from leakage-safe monthly trajectories, combined with past-only structural-change detection and robust multi-fold temporal selection, can capture new information beyond Experiment 12 and improve decision-useful early/mid forecasting without sacrificing future generalization.

## 1. Learned latent trajectory regimes

V2 fits a training-only Gaussian-mixture model over leakage-safe Experiment 12 trajectory dimensions including:

- 3/6/12-month cost growth and acceleration;
- 3/6/12-month expenditure velocity and acceleration;
- 3/6/12-month schedule-slippage velocity and acceleration;
- spend-vs-expected-progress gap;
- cost/schedule revision magnitude;
- cost/schedule worsening streaks.

The regime model is unsupervised: it does not receive final cost-overrun or final delay targets.

It produces soft state information rather than a single brittle label:

- probability of each learned regime;
- most likely regime;
- regime confidence;
- regime entropy;
- regime surprise / negative log likelihood.

Every rolling validation fold fits its own regime encoder using only that fold's fitting years. The final encoder is fit only on the full training window. Future holdout projects are transformed with the frozen encoder and never participate in regime learning.

## 2. Past-only structural-change detection

V2 replaces short-vs-long heuristic turning proxies with an online two-sided CUSUM detector for:

- cost trajectory;
- schedule-slippage trajectory;
- expenditure trajectory.

At report `t`, the CUSUM baseline is calculated only from earlier reports for the same canonical project. The detector records:

- current change score;
- structural-change event indicator;
- reports since the most recent detected change;
- synchronized cost/schedule/spend change intensity.

Appending a future report must not alter an earlier feature vector; tests enforce this as-of property.

## 3. Multi-fold temporal model selection

V1 selected a feature group using only the final internal validation block, allowing a small local gain to be mistaken for a stable temporal effect.

V2 uses up to three rolling forward-only folds inside the training period. Each fold obeys:

`fit years < validation years < untouched final holdout`

Candidate groups are:

1. `stage_weighted_production`
2. `learned_regimes`
3. `learned_regimes_plus_change_points`

A learned-regime group is accepted only if it satisfies all of the following relative to the stage-weighted production-feature challenger:

- at least **0.25% mean improvement** in the stage-priority selection objective;
- wins at least two folds, or all-but-one when fewer folds are available;
- no validation fold worse than **-1.5%**.

If those conditions are not met, v2 falls back to the stage-weighted production feature set instead of accepting a one-window improvement.

## 4. Early/mid lifecycle optimization

Production evaluation remains project-balanced so headline numbers are directly comparable.

Challenger training, however, deliberately gives more importance to stages where an intervention is still useful:

- early: 2.5x training multiplier;
- early-mid / mid: 2.1x;
- late-mid / late: 1.25x;
- very-late: 0.70x.

The temporal selection objective combines overall project-balanced MAE with a lifecycle-priority MAE that weights early and mid stages more strongly. This prevents thousands of very-late snapshots from hiding a deterioration in early-warning performance.

Both ordinary overall MAE and stage-level/stage-balanced MAE remain reported on the untouched future cohort.

## Fair comparison against promoted Experiment 12 production

The current production model remains frozen.

Experiment 13 v2:

- starts from the exact promoted Experiment 12 production feature contract;
- retains the production-selected regressor family for cost and delay;
- adds learned-regime/change-point context only inside the challenger;
- uses the same temporal project split and comparable future cohort;
- evaluates production and challenger with the same project-balanced holdout weights;
- writes only namespaced experiment artifacts;
- has `promotion_allowed: false`.

## Leakage policy

For every snapshot, raw trajectory and change-point signals use only current/earlier official reports for that canonical project.

For learned regimes:

- each internal fold fits its imputer, scaler and Gaussian mixture only on that fold's fitting period;
- validation rows are transformed by the frozen fold encoder;
- the final regime encoder is fit only on the full requested training window;
- final future holdout rows never affect regime discovery, feature selection or model fitting.

## Evaluation contract

Run both standard controlled windows:

- training 2001–2019, future holdout through the latest valid completion year;
- training 2001–2021, future holdout through the latest valid completion year.

For cost and delay independently report:

- production MAE;
- Experiment 13 v2 MAE;
- absolute and percentage MAE improvement;
- paired-project bootstrap probability candidate is better;
- paired-project 95% improvement interval;
- lifecycle-stage MAE;
- stage-balanced MAE;
- comparable project/snapshot counts;
- rolling-fold selection diagnostics;
- selected v2 feature group.

A green workflow means technical success only. Promotion requires reproducibly positive scientific evidence.

## Neural-network follow-up

A neural sequence model is intentionally **not** mixed into this PR. Doing so would simultaneously change trajectory representation, validation policy, weighting and model family, making attribution impossible.

If v2 still fails to improve, a separate controlled neural challenger may be justified only if it consumes the raw ordered monthly sequence (for example a compact GRU/TCN/Transformer-style encoder with static project covariates), rather than an MLP over the same engineered tabular features.

That neural experiment must compare against the same promoted production baseline and use the same future holdout contract.

## Promotion rule

There is no automatic promotion.

Cost and delay are judged independently. A gain on one target does not justify promoting a regression on the other. Any accepted component requires positive results on both required temporal windows and a separate deliberate production PR.
