# Project explanation publication

Project Detail reads project-level model evidence from a versioned publication artifact. A normal historical HTTP request never loads the frozen model bundle and never computes SHAP.

The lifecycle is:

1. training publishes an immutable production bundle and `prediction_validation.csv`;
2. the explanation publisher reconstructs the exact ledger snapshot;
3. Cost, Delay, and Risk must all reproduce the ledger before explanation begins;
4. wrapper-level, deterministic two-path local contributions are checked for additivity against each final output;
5. snapshot-bounded Operational Drivers are evaluated independently from official PAIMANA history;
6. `project_explanations.jsonl` and its SHA-256 metadata are atomically published;
7. saved portfolio views embed compact factors/statuses and fingerprint the explanation artifact;
8. FastAPI loads an identity-sensitive in-memory index for effectively constant-time lookups.

The method is a deterministic local Shapley-style path approximation over the complete serialized predictor, not TreeSHAP over an inner estimator. Cost and Delay explain the final numeric output. Risk explains the predicted-class probability. The artifact retains all feature contributions for reconstruction and exposes the five largest contributions to the UI.

## Publish frozen explanations

This operation uses existing models only and does not retrain them:

```bash
.venv/bin/python scripts/build_frozen_local_shap.py --window 2001_2021 --all --resume
.venv/bin/python scripts/build_frozen_local_shap.py --all-windows --resume
```

`--all-windows` currently means the published monthly-lifecycle bundles
`2001_2021` and `2001_2022`. The legacy `2001_2017` saved view is intentionally
outside this explanation publication scope and continues to report explanations
as unavailable.

Useful controls are `--project <code>`, `--limit <N>`, `--force`, and `--report <path>`. A separate fsynced build journal supports `--resume`; the last canonical artifact remains untouched until atomic replacement. Entries are invalidated when the signed model, metadata, ledger, run, or dataset identity changes.

After publication, rebuild the saved project view with:

```bash
.venv/bin/python -c "from backend.app.services.range_portfolio_service import write_saved_window_view; write_saved_window_view('2001_2021')"
```

## Verify coverage

```bash
.venv/bin/python scripts/verify_project_explanation_coverage.py
```

The validator compares the exact Projects UI cohort with the indexed publication and fails non-zero for missing, stale, incomplete, or reproduction-failed records. It writes `test-output/project-explanation-coverage.json` by default. An empty `operational_drivers: []` is valid evidence that no material rule fired; a missing field or unavailable status is not.

## Interpretation boundary

Model evidence describes how inputs moved a model output relative to a deterministic reference. Operational Drivers are observed warning signals derived from official PAIMANA records. Neither is causal proof. The system does not infer land acquisition, contractor, clearance, litigation, or funding causes because the authorised source data does not contain those fields.
