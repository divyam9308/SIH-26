# Experiment 34 production promotion — Delay only

## Promotion decision

Experiment 34 is promoted to production for **Delay only** on the selected production contract.

- Training window: **2001–2021**
- Future holdout: **2022–2025**
- Official production-comparable evaluation cohort: **721 projects / 11,200 snapshots**
- Previous production Delay MAE: **534.691 days**
- Exp34 Delay MAE: **501.303 days**
- Delay improvement: **6.2444%**
- Exp34 blend: **20% ExtraTrees / 60% LightGBM / 20% XGBoost**

## Unchanged targets

- Cost remains the promoted Experiment 12 trajectory production model.
- Risk remains the existing production classifier.
- The Delay promotion code guards Cost and Risk artifacts against accidental replacement.

## Delay production path

Production Delay uses Experiment 34's causal full-lifecycle path features and rolling out-of-fold ensemble. Path features use only the current or prior official snapshot history for each project. Ensemble weights are selected only from rolling validation folds inside the training period; the future holdout is not used for selection.

## Production evaluation contract

The official headline production Delay MAE uses the same exact 721-project comparable cohort as the promoted Cost contract. The broader 728-project holdout remains available only as a diagnostic; its Exp34 Delay MAE is 503.555 days.

For the selected 2001–2021 production run, CI refuses promotion if the cohort is not exactly 721 projects / 11,200 snapshots or if the reproduced Delay MAE differs from 501.303 days. The production Cost MAE must remain 26.872.
