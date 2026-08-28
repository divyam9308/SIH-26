# Exp25 current-production retest

This branch does not promote Exp25. It retests the Exp25 feature idea against the current production stack after Exp34 Delay promotion.

- Production Cost: Exp12 trajectory model.
- Production Delay: Exp34 path-dependence + rolling-OOF blend.
- Evaluation: shared Exp12-comparable cohort for Cost and Delay; for 2001-2021 this must be exactly 721 projects / 11,200 snapshots.
- Exp25 candidate additions: reusable project-name semantics, structured PAIMANA context, and causal milestone trajectory features.
- Raw project name is never passed to a model.
- Cost keeps the production-selected family fixed.
- Delay keeps the Exp34 three-family architecture and current training-only OOF blend weights fixed; only the Exp25 feature group changes.
- Feature-group selection uses only an internal forward split within the training window.
- No future holdout data is used to select features, models, or weights.
- The PR must remain an isolated challenger until both 2001-2019 and 2001-2021 Actions results are reviewed.
