# Experiment 2: hybrid cost-regime mixture-of-experts

## Purpose

Experiment 2 tests whether cost-overrun prediction improves by first classifying a project into a cost regime and then using regime-specific regressors. It is isolated from production and does not activate or overwrite any production model.

## Controlled comparison

The GitHub Actions run used the same official PAIMANA completed-project cohort and five-feature production contract as the registered `2001_2017` baseline:

- training years: 2001-2017
- testing years: 2018-2024
- training projects: 303
- testing projects: 156
- seed: 26103
- features: `approved_cost_cr`, `sector_average_delay`, `sector_average_cost_overrun`, `sector`, `project_size_category`
- baseline cost MAE reproduced exactly at 32.886 percentage points
- baseline RMSE reproduced at 60.183
- baseline R2 reproduced at -0.0186

No `project_history.csv`, synthetic lifecycle data, monthly extraction, future outcome feature, delay-model change, risk-model change, production registry change, or production model overwrite was used.

## Regimes

The thresholds were fixed before final holdout evaluation:

- COST_SAVING: <= 0%
- LOW: 0% to 20%
- MEDIUM: 20% to 100%
- HIGH: 100% to 200%
- EXTREME: > 200%

Training distribution:

| Regime | Projects | Share |
| --- | ---: | ---: |
| COST_SAVING | 188 | 62.05% |
| LOW | 37 | 12.21% |
| MEDIUM | 47 | 15.51% |
| HIGH | 17 | 5.61% |
| EXTREME | 14 | 4.62% |

All regimes exceeded the pre-declared 12-project minimum, so each received its own CatBoost regressor.

## Classifier

The final holdout classifier achieved:

- accuracy: 0.6218
- balanced accuracy: 0.2994
- macro precision: 0.4619
- macro recall: 0.2994
- macro F1: 0.3166
- weighted F1: 0.6176
- naive majority-class macro F1: 0.1645
- mean maximum class probability: 0.4632
- mean normalized entropy: 0.7692

Per-class F1:

- COST_SAVING: 0.7854
- LOW: 0.2143
- MEDIUM: 0.2500
- HIGH: 0.3333
- EXTREME: 0.0000

The classifier therefore contains more signal than a majority-class classifier overall, but it is weak outside the dominant COST_SAVING class and did not correctly identify an EXTREME holdout project. The latest-year internal temporal classifier check is also not broadly representative because all nine 2017 validation outcomes belong to COST_SAVING; its majority baseline score is therefore not useful as a multiclass quality estimate.

## Routing

Two strategies were evaluated after architecture decisions were fixed:

1. **Hard routing**: choose the highest-probability regime expert, with a confidence-derived blend toward the global fallback.
2. **Soft mixture**: probability-weight all regime expert predictions, then blend toward the global fallback when classifier confidence is weak.

The blend factor is pre-declared as `clip((max_class_probability - 0.2) / 0.8, 0, 1)` and was not tuned on the final holdout.

## Measured results

| Metric | Production baseline | Hard routing | Soft mixture |
| --- | ---: | ---: | ---: |
| MAE | 32.886 | 33.539 | 33.344 |
| RMSE | 60.183 | 59.758 | 58.834 |
| Median absolute error | 16.894 | 20.467 | 19.068 |
| R2 | -0.0186 | -0.0043 | 0.0265 |
| MAPE | 257.604 | 274.565 | 133.667 |

The best Experiment 2 variant by the pre-declared primary metric, MAE, was **soft mixture** at **33.344 pp**.

Compared with the production baseline:

- absolute MAE change: **+0.458 pp worse**
- relative MAE change: **1.39% worse**
- decision category: **WORSE**

Soft routing did improve RMSE by 1.349 points, moved R2 from -0.0186 to 0.0265, and substantially lowered MAPE. However, the primary metric was defined before the experiment as holdout MAE, and MAE degraded. Median absolute error also worsened by 2.174 pp.

## Error distribution

Baseline versus soft mixture:

- within 10 pp: 44 vs 38 projects
- within 20 pp: 87 vs 80
- within 30 pp: 108 vs 101
- over 30 pp: 48 vs 55
- p90 absolute error: 71.469 vs 62.264
- p95 absolute error: 110.055 vs 88.802
- maximum error: 362.992 vs 356.374

The mixture reduces the upper error tail but makes more ordinary predictions less accurate. That explains why RMSE and tail percentiles improve while overall MAE worsens.

## Extreme-project behavior

The hybrid model still fails to identify the most important extreme cases reliably. For example, the largest-error holdout project had an actual overrun of 354.61%, while the classifier assigned only 2.57% probability to EXTREME and selected MEDIUM. Its hybrid prediction remained -1.77%, improving the baseline error only slightly from 362.99 pp to 356.37 pp.

This is consistent with the final EXTREME recall of 0.0 and indicates that the current five-feature public PAIMANA contract does not provide enough information for reliable extreme-regime routing.

## Leakage and production safety

- Actual overrun creates training labels only and is never an inference feature.
- Actual regime is never used to select an expert during inference.
- Historical priors follow the same production past-only policy.
- The experiment reproduced the baseline before comparison.
- Production cost, delay and risk artifacts and the active registry were SHA-256 checked before and after execution and remained unchanged.

## Decision

**Do not replace the current production cost model with Experiment 2.**

Experiment 2 is scientifically useful because it shows that a regime classifier has some predictive signal and can reduce large-error tail magnitude, but the mixture-of-experts architecture does not improve the pre-declared primary metric. The best hybrid MAE is 33.344 compared with the production 32.886, a 1.39% degradation.

Measured on GitHub Actions run `32605890098` using `ubuntu-latest`. Detailed generated JSON reports and prediction rows are preserved in the `experiment-2-results` workflow artifact.