# InfraSight AI — SIH26103

**Predictive infrastructure project intelligence for MoSPI / PAIMANA.**

InfraSight AI is an end-to-end prototype for **Smart India Hackathon 2026 problem statement SIH26103**, “Use case on web-based integrated project-monitoring platform”. The system turns public PAIMANA project records into a modular decision-support workspace for cost/schedule overrun intelligence, project prioritisation, explainability, peer benchmarking, historical replay, data-quality review and scenario sensitivity analysis.

> **Data integrity:** the showcased project rows are real PAIMANA public records. The repository deliberately excludes high-value records when an official project code was not surfaced; it does not invent PAIMANA identifiers.

## What is implemented

- **Portfolio command dashboard** over the real curated PAIMANA subset.
- **Project explorer** with sector/search filters and source links.
- **Official archive ingestion** with immutable PDFs, SHA-256 manifesting and nullable normalization.
- **Real longitudinal archive table** with 4,692 project-month observations across 1,844 official project codes.
- **Leakage-safe temporal features and labels** with project-level, time-based cohorts.
- **Cost and delay regression** across Random Forest, XGBoost and CatBoost with automatic selection.
- **Project-level SHAP explanations** from the selected forecasting artifacts.
- **Historical-cutoff verification** using older completed projects for fitting and 2025–26 completions for evaluation.
- **Project Forecast, Model Performance and Prediction Accuracy** judging flows.
- **Model Simulation** with real PAIMANA completed-project windows: 2001–15 → 2016–21 and 2015–21 → 2022–24 (2025–28 stay forecast-only until official outcomes are published).
- **Priority / intervention queue** combining model risk signals with financial exposure.
- **Peer benchmarking** against similar projects in the same sector.
- **Historical Time Machine** using real monthly PAIMANA/Flash Report snapshots.
- **Scenario Explorer** for sensitivity testing; outputs are explicitly not presented as causal guarantees.
- **Data Quality Observatory** that surfaces missing/contradictory operational fields rather than silently repairing them.
- **Grounded analytics assistant** that answers from computed local portfolio analytics without generating risk numbers through an LLM.
- **FastAPI API + modular browser UI**, with feature/page files separated in the style of the companion ATS project.

## Real dataset currently included

The reproducible dataset seed builds:

- **96** PAIMANA May 2026 project rows with surfaced official project codes.
- **14** official historical snapshots across selected high-value projects.
- **6** original 2024–25 PAIMANA archive PDFs retained unchanged with hashes.
- **4,692** normalized official archive observations across **1,844** project codes; **1,490** projects have multiple snapshots.
- **11** represented sectors.
- roughly **₹5.67 lakh crore** of original approved project cost in the included rows.

Primary source surfaces:

- PAIMANA Public Dashboard: `https://ipm.mospi.gov.in/Home/PublicDashboard`
- PAIMANA high-value project surface: `https://ipm.mospi.gov.in/Home/GetHighlyValue`
- PAIMANA / MoSPI monthly Flash Reports, including March 2026.
- PAIMANA Project Monitoring Archive: `https://paimana-proj.mospi.gov.in/ReportPage/ArchiveProjectMonitoring`

See [`docs/data_pipeline.md`](docs/data_pipeline.md) and [`docs/data-provenance.md`](docs/data-provenance.md) for extraction details and limitations.

## SIH forecasting demo

The judging flow is available at **Project Forecast**. It selects a project, loads its most recent longitudinal snapshot, and shows predicted cost escalation, delay, risk level, and feature-level SHAP factors through `GET /api/projects/{project_id}/forecast`.

`data/project_history.csv` is a deterministic synthetic monthly demonstration dataset, documented in [`docs/data_source.md`](docs/data_source.md). It demonstrates the replaceable PAIMANA/OCMS-compatible schema and must be replaced with an authorised monthly export before operational use. Train it with:

```bash
python scripts/generate_project_history.py  # demo data only
python -m backend.app.ml.train
```

## Current model results

Model selection uses a project-level time split: training projects started through 2023, validation projects in 2024–25, and test projects from 2026. The latest test-cohort results are recorded in `models/model_metrics.json`; negative R² values remain visible instead of being hidden.

The separate cutoff backtest is stricter: verification models fit 25 projects completed through 2024, then evaluate 20 unseen projects completed in 2025–26.

| Backtest target | MAE | RMSE | R² | Mean accuracy |
|---|---:|---:|---:|---:|
| Final cost overrun | 2.332 pp | 3.418 pp | 0.8234 | 94.2% |
| Final delay | 20.957 days | 25.640 days | 0.8614 | 93.63% |

Elevated-risk classification achieves 85.0% accuracy and 0.8571 F1 on that held-out cohort. These metrics use the documented synthetic completion trajectories, not the official archive rows.

### Forecasting boundary

The archive ingestion and longitudinal monitoring observations are real. PAIMANA's public ongoing-project reports do not consistently publish project-level final actual cost and actual completion, so the repository does not fabricate those labels. The bundled final-outcome models therefore remain a **demonstration trained on deterministic synthetic completion trajectories**. Replace them with an authorized PAIMANA/OCMS completed-project export before operational use.

## Real historical model simulation

The **Model Simulation** page is separate from the older demonstration forecast. It uses only `data/processed/paimana_completed_outcomes.csv`, extracted from official PAIMANA completed-project archive tables. It never reads `data/project_history.csv`.

```bash
PYTHONPATH=. python scripts/ingest_paimana_completed_reports.py --from-year 2001 --to-year 2025
PYTHONPATH=. python train.py --start-year 2001 --end-year 2015
PYTHONPATH=. python train.py --start-year 2015 --end-year 2021
PYTHONPATH=. python evaluate_model.py --model 2001_2015
PYTHONPATH=. python evaluate_model.py --model 2015_2021
```

Reported completion expenditure and completion month are targets only; they never become model inputs. The V2 reliability report only scores the official outcomes available through 2024. It excludes 2025–2028 from metrics until PAIMANA publishes recorded completion outcomes.

## Official monthly lifecycle forecasting

The production monthly upgrade discovers all official Flash Reports from 2001–02 through 2024–25, preserves immutable PDFs, uses detected-layout parsers, constructs exact/audited project trajectories, and trains snapshot-at-T models without synthetic history.

```bash
# Official index only
PYTHONPATH=. python scripts/build_monthly_ml_pipeline.py --discover-only

# Download, parse, resolve identity, build trajectories and datasets
PYTHONPATH=. python scripts/build_monthly_ml_pipeline.py

# Reuse cached PDFs and train/evaluate both required windows plus ablations/SHAP
PYTHONPATH=. python scripts/build_monthly_ml_pipeline.py --local-only --train

# One dynamic window
PYTHONPATH=. python scripts/build_monthly_ml_pipeline.py --local-only --train \
  --training-start 2001 --training-end 2015 --test-end 2021
```

The official processed monthly snapshot dataset is committed at `data/processed/paimana_monthly_snapshots.csv` so a normal clone can retrain immediately. Raw PDFs, parser caches and model binaries remain reproducible/ignored. Refreshing the official dataset is an occasional data operation; clicking **Retrain Lifecycle Models Live** only loads this processed file and trains the selected window. Versioned metadata, comparison reports, ingestion audits and the human-readable `reports/monthly_lifecycle_upgrade_report.md` capture exact evidence and limitations.

## Architecture

```text
SIH-26/
├── backend/
│   └── app/
│       ├── core/                  # configuration
│       ├── ml/                    # features, training, forward labels
│       ├── routes/                # thin API routes by feature
│       └── services/              # data, prediction, SHAP, peers, history...
├── frontend/
│   └── src/
│       ├── components/            # shared UI components
│       ├── features/              # feature-specific view modules
│       ├── pages/                 # one top-level file per screen
│       ├── services/              # API client
│       ├── styles/                # base/layout/components/page styles
│       └── utils/                 # formatting helpers
├── data/
│   ├── raw/                       # generated source-aligned real records
│   └── processed/                 # generated engineered model dataset
├── models/                        # generated binaries/metrics + retained training record
├── scripts/                       # seed/train/run entrypoints
├── tests/                         # data, API, model and browser tests
└── docs/                          # architecture/methodology/provenance
```

The organization mirrors the feature/page/service separation used in `fyndbridge-ats`, but intentionally avoids putting an entire feature into one huge page/controller file.

## Run locally

### Full-stack frontend development

`backend/` remains the authoritative FastAPI and ML implementation. The React
application in `frontend/` is its only user interface and consumes it over
`/api`; it does not contain model logic or model results.

Start the API in one terminal:

```bash
PORT=8000 ./scripts/run_local.sh
```

Then start the Vite frontend in another:

```bash
cd frontend
npm ci
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`, so no development CORS rule is
needed. `VITE_API_BASE_URL` may be set in `frontend/.env.local` when deploying
against a different API origin; see `frontend/.env.example`. For FastAPI to
serve the production UI itself, run `npm run build` in `frontend/` first; the
backend serves `frontend/dist` and supports nested SPA routes.

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

./scripts/run_local.sh          # rebuilds real-data CSVs and trains missing model artifacts automatically
```

To refresh official archive inputs explicitly:

```bash
python scripts/ingest_paimana_archive.py
python scripts/ingest_paimana_archive.py --local-only  # reproducible offline normalization
```

Open:

```text
http://127.0.0.1:8000
```

The frontend has no CDN/runtime dependency; FastAPI serves the API and the modular SPA together. On a fresh clone, first launch rebuilds the real PAIMANA dataset from the checked-in source-aligned seed and trains the selected inference artifacts before starting the server.

## Run tests

```bash
pytest
```

For the browser smoke test:

```bash
pip install -r requirements-dev.txt
playwright install chromium
python tests/browser_smoke.py
```

The browser smoke covers the primary judging path across **Dashboard → Project Forecast → Model Performance → Prediction Accuracy**, plus project/history/scenario views.

## Real test case: Rajasthan Refinery (`701263`)

From the official PAIMANA row:

- Original cost: **₹43,129 Cr**
- Revised cost: **₹79,459 Cr**
- Cumulative expenditure: **₹69,997 Cr**
- Original completion: **31 Oct 2022**
- Revised completion: **30 Jun 2026**
- Physical progress: **92%**
- Observed cost escalation: **84.2%**
- Observed schedule extension: **1,338 days / ~3.7 years**

The Forecast screen keeps those observed facts separate from the synthetic-demo future forecast and labels the model scope directly in the UI.

## SIH26103 mapping

| Problem-statement outcome | InfraSight module |
|---|---|
| Cost Overrun Prediction Model | Temporal XGBoost/Random Forest/CatBoost regression pipeline |
| Time Overrun Prediction Model | Temporal XGBoost/Random Forest/CatBoost regression pipeline |
| Project Risk Scoring Framework | Portfolio review-priority engine |
| Early Warning Alert System | Early Warnings queue |
| Benchmarking & Comparative Analytics | Sector peer benchmarking |
| Cost Escalation Driver Analysis | Local SHAP driver view from selected forecast models |
| AI-powered Monitoring Dashboard | Dashboard + project intelligence pages |
| LLM-enabled Project Intelligence Assistant | Grounded analytics interface now; optional LLM adapter can be added after core analytics validation |
| Documentation & reproducibility | This repository + docs + tests |

## Why the code avoids overclaiming

- SHAP is described as **feature contribution**, not causality.
- Scenario output is described as **model sensitivity**, not a guaranteed intervention effect.
- Missing PAIMANA values are surfaced as data-quality signals.
- Current baseline models are not called forward-validated future forecasts.
- No risk number is generated by an LLM.
- Every showcased project retains its official source URL.

## Operationalization milestone

Obtain an authorized completed-project PAIMANA/OCMS export containing final actual cost and actual completion dates. Feed it through the existing normalized schema, then rerun the temporal training and cutoff verification before presenting the model as production evidence.
