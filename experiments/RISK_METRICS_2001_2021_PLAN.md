# Risk macro-metrics experiment — 2001–2021

Objective: improve macro F1 / precision / recall for the existing four-class lifecycle risk classifier without tuning on the frozen 2022–2025 holdout.

Method:
- Keep the current risk feature contract and four classes: LOW, MEDIUM, HIGH, CRITICAL.
- Generate temporal rolling out-of-fold class probabilities using only historical training data.
- Search class weights and class-specific decision offsets/thresholds on OOF predictions only.
- Optimize a predeclared macro-F1 objective subject to precision and recall non-regression constraints.
- Freeze selected parameters before evaluating on 2022–2025.

Promotion gate:
- Macro F1 improves versus baseline.
- Macro precision and macro recall must not materially regress; report all three.
- No threshold/class-weight selection may observe 2022–2025 labels.
- No changes to risk labels or holdout membership.
