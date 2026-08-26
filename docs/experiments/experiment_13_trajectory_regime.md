# Experiment 13 — Advanced trajectory/regime interaction model

## Status

**EXPERIMENTAL / PENDING EVIDENCE**

This PR installs Experiment 13 as the active challenger. It does **not** promote or modify production.

Current production remains:

- cost: promoted Experiment 12 trajectory-enhanced cost baseline;
- delay: existing production lifecycle delay model;
- risk: existing production lifecycle risk model.

## Hypothesis

Experiment 12 proved that past-only monthly trajectory information can improve cost-overrun forecasting. Experiment 13 tests the next question:

> Does the meaning of a trajectory signal change depending on the project's current deterioration/recovery state, lifecycle stage, cross-signal context, recent turning points and regime transitions?

The experiment keeps the current production model family, temporal split and weighting contract fixed. The changed dimension is the **trajectory representation**.

## Added signal families

### 1. Continuous project-regime / pressure scores

Rather than forcing each project into one brittle hard class, Experiment 13 derives continuous past-only scores for:

- cost pressure;
- schedule pressure;
- execution/spend stall pressure;
- recovery strength;
- combined cost + schedule pressure;
- cost-vs-schedule pressure imbalance;
- revision volatility.

These scores are built from already leakage-safe Experiment 12 history signals such as normalized 3/6/12-month growth, acceleration, revision magnitude and worsening streaks.

### 2. Cross-signal interactions

Experiment 13 explicitly represents compound deterioration, including:

- cost pressure × schedule pressure;
- cost pressure × execution stall;
- schedule pressure × execution stall;
- cost acceleration × schedule acceleration;
- cost worsening streak × schedule worsening streak.

Tree models can discover interactions implicitly, but this experiment tests whether a small, domain-motivated interaction set improves temporal generalization without changing model family.

### 3. Lifecycle-conditioned trajectory interactions

The same trajectory may have different meaning early versus late in project execution. Experiment 13 therefore adds:

- cost pressure × lifecycle progress;
- schedule pressure × lifecycle progress;
- combined pressure × lifecycle progress;
- cost growth × lifecycle progress;
- schedule acceleration × lifecycle progress;
- spend-vs-expected-progress gap × lifecycle progress.

No future completion outcome is used to create these features.

### 4. Turning-point / structural-change signals

Experiment 13 compares short-horizon and long-horizon trajectory behavior to detect recent changes in direction or intensity:

- 3-month vs 12-month cost-growth divergence;
- 3-month vs 12-month schedule-slippage divergence;
- 3-month vs 12-month spend-velocity divergence;
- cost, schedule and spend turning strength;
- synchronized cost + schedule turning strength.

These are deterministic as-of features; they do not use the final holdout to learn change-point thresholds.

### 5. Regime-transition signals

Experiment 13 also models how pressure itself is changing:

- 3-month cost-pressure velocity;
- 3-month schedule-pressure velocity;
- 3-month compound-pressure velocity;
- worsening-transition strength;
- recovery-transition strength;
- consecutive worsening-regime streak.

This distinguishes a project that has been consistently problematic from one that has only recently started deteriorating rapidly.

## Target-specific ablation

Experiment 13 does not force all new features into either target.

For **cost** and **delay separately**, it evaluates these feature depths on an internal temporal validation block inside the training window:

1. `production_only`
2. `regime_scores`
3. `regime_plus_interactions`
4. `regime_interactions_turning`
5. `all_regime_context`

The lowest-MAE group is selected before fitting on the full training window.

The future holdout is never used to select the feature group.

This also means Experiment 13 can legitimately select `production_only` for one target if its new context does not validate.

## Fair comparison against promoted Experiment 12 production

Experiment 13 starts from the exact promoted production feature contract:

- the production cost model receives its existing Experiment 12-selected trajectory features;
- the production delay model receives its current production delay features;
- Experiment 13 adds only its new regime/context representation on top;
- the production-selected regressor family is retained for each target;
- the production/challenger comparison uses the same frozen dataset, temporal split and comparable project cohort.

This prevents an apparent Experiment 13 gain from actually being caused by silently reverting or changing Experiment 12.

## Leakage policy

For snapshot `t`, every Experiment 13 feature is derived only from:

- the current official snapshot;
- earlier official snapshots for the same canonical project.

Appending an extreme future report must not alter an earlier Experiment 13 feature vector. Tests enforce this property.

Feature-group selection uses only an internal historical block within the requested training window. Holdout outcomes are untouched until final evaluation/reveal.

## Evaluation contract

The experiment should be audited on at least both standard windows:

- training: 2001–2019, future holdout through the latest valid completion year;
- training: 2001–2021, future holdout through the latest valid completion year.

Report independently for cost and delay:

- production MAE;
- Experiment 13 MAE;
- absolute MAE reduction;
- relative improvement percentage;
- paired-project bootstrap probability candidate is better;
- paired-project 95% improvement interval;
- lifecycle-stage MAE;
- stage-balanced MAE;
- comparable project/snapshot counts;
- selected internal feature group.

A green workflow means only that the experiment executed successfully. It does **not** mean the model improved.

## Promotion rule

There is no automatic promotion.

Cost and delay are judged independently. A strong cost result does not justify promoting a regressed delay model, and vice versa.

Promotion should require a separate deliberate PR after the result is reproducibly positive across the required temporal windows and provenance checks.

## Retrain & Compare integration

`backend/app/ml/experiments/adapter_exp13.py` registers:

- `EXPERIMENT_ID = exp_13`
- sequence: `13`
- scope: `cost_delay`

The generic Model Simulation workflow will therefore use Experiment 13 as the active challenger while this adapter is installed.

Experiment artifacts remain isolated under the generic experiment artifact namespace and have `promotion_allowed: false`.
