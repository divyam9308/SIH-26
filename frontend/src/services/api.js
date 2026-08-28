const JSON_HEADERS = { "Content-Type": "application/json" };
const DEFAULT_TIMEOUT_MS = 45000;
const HEAVY_TIMEOUT_MS = 30 * 60 * 1000;
const COMPARE_TIMEOUT_MS = 60 * 60 * 1000;
const SELECTED_MODEL_KEY = 'selected_validation_model';
const ACTIVE_LIFECYCLE_KEY = 'active_lifecycle_run';
const DEFAULT_PRODUCTION_LIFECYCLE_WINDOW = '2001_2021';

async function request(path, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    const type = response.headers.get('content-type') || '';
    if (!response.ok || !type.includes('application/json')) {
      let detail = `Request failed (${response.status})`;
      if (type.includes('application/json')) {
        try { detail = (await response.json()).detail || detail; } catch (_) {}
      }
      throw new Error(detail);
    }
    return response.json();
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw new Error('Request timed out before the backend returned the completed model run. The current page has discarded the pending session; retry the requested range.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

const selectedModel = () => sessionStorage.getItem(SELECTED_MODEL_KEY) || '';
const withModel = (path, model = selectedModel()) => model ? `${path}${path.includes('?') ? '&' : '?'}model=${encodeURIComponent(model)}` : path;

function getActiveLifecycleRun() {
  const raw = sessionStorage.getItem(ACTIVE_LIFECYCLE_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch (_) { return null; }
}

function setActiveLifecycleRun(run) {
  if (!run) {
    sessionStorage.removeItem(ACTIVE_LIFECYCLE_KEY);
    return;
  }
  sessionStorage.setItem(ACTIVE_LIFECYCLE_KEY, JSON.stringify(run));
  if (run.window) sessionStorage.setItem(SELECTED_MODEL_KEY, run.window);
}

export const api = {
  health: () => request('/api/health'),
  portfolioSummary: () => request('/api/portfolio/summary'),
  portfolioRisk: (limit = 20) => request(`/api/portfolio/risk?limit=${limit}`),
  projects: ({ search = '', sector = '', limit = 100 } = {}) => {
    const p = new URLSearchParams({ limit: String(limit) });
    if (search) p.set('search', search);
    if (sector) p.set('sector', sector);
    return request(`/api/projects?${p.toString()}`);
  },
  project: (code) => request(`/api/projects/${encodeURIComponent(code)}`),
  prediction: (code) => request(`/api/projects/${encodeURIComponent(code)}/prediction`),
  forecast: (code) => request(`/api/projects/${encodeURIComponent(code)}/forecast`),
  peers: (code) => request(`/api/projects/${encodeURIComponent(code)}/peers`),
  modelMetrics: () => request('/api/models/metrics'),
  modelImportance: () => request('/api/models/importance'),
  lifecycleRuns: () => request('/api/models/lifecycle-runs'),
  monthlyLifecycleComparison: () => request('/api/models/monthly-lifecycle-comparison'),
  lifecycleForecast: (code, window = DEFAULT_PRODUCTION_LIFECYCLE_WINDOW) => request(`/api/projects/${encodeURIComponent(code)}/lifecycle-forecast?window=${encodeURIComponent(window)}`),
  lifecycleEvolution: (projectId, window = DEFAULT_PRODUCTION_LIFECYCLE_WINDOW) => request(`/api/models/monthly-lifecycle-evolution/${encodeURIComponent(projectId)}?window=${encodeURIComponent(window)}`),
  validationReport: (modelVersion = selectedModel()) => request(withModel('/api/models/validation', modelVersion)),
  predictionValidation: (limit = 100, modelVersion = selectedModel()) => request(withModel(`/api/models/prediction-validation?limit=${limit}`, modelVersion)),
  rollingValidation: (modelVersion = selectedModel()) => request(withModel('/api/models/rolling-validation', modelVersion)),
  retrainModel: (startYear, endYear) => request('/api/models/retrain', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ start_year: Number(startYear), end_year: Number(endYear) }) }, HEAVY_TIMEOUT_MS),
  residualOverrunExperiment: (startYear, endYear) => request('/api/models/experiments/residual-overrun', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ start_year: Number(startYear), end_year: Number(endYear) }) }, HEAVY_TIMEOUT_MS),
  setValidationModel: (model) => model ? sessionStorage.setItem(SELECTED_MODEL_KEY, model) : sessionStorage.removeItem(SELECTED_MODEL_KEY),
  getValidationModel: () => selectedModel(),
  setActiveLifecycleRun,
  getActiveLifecycleRun,
  clearActiveLifecycleRun: () => {
    sessionStorage.removeItem(ACTIVE_LIFECYCLE_KEY);
    sessionStorage.removeItem(SELECTED_MODEL_KEY);
  },
  simulationVersions: () => request('/api/model-simulations', {}, 90000),
  runSimulation: (version) => request(`/api/model-simulations/${encodeURIComponent(version)}/run`, { method: 'POST' }),
  trainCustomSimulation: (startYear, endYear, runId = null) => request('/api/model-simulations/custom/train', {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ start_year: Number(startYear), end_year: Number(endYear), run_id: runId }),
  }, HEAVY_TIMEOUT_MS),
  retrainAndCompare: (startYear, endYear, experimentId = null) => request('/api/model-simulations/custom/retrain-compare', {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ start_year: Number(startYear), end_year: Number(endYear), experiment_id: experimentId }),
  }, COMPARE_TIMEOUT_MS),
  customSimulationProjects: (sessionId, year) => request(`/api/model-simulations/custom/${encodeURIComponent(sessionId)}/projects?year=${encodeURIComponent(year)}`),
  comparisonProjects: (sessionId, year) => request(`/api/model-simulations/compare/${encodeURIComponent(sessionId)}/projects?year=${encodeURIComponent(year)}`),
  predictCustomSimulation: (sessionId, recordIndex) => request(`/api/model-simulations/custom/${encodeURIComponent(sessionId)}/predict`, {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ record_index: Number(recordIndex) }),
  }),
  predictComparison: (sessionId, recordIndex) => request(`/api/model-simulations/compare/${encodeURIComponent(sessionId)}/predict`, {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ record_index: Number(recordIndex) }),
  }),
  revealCustomSimulation: (sessionId, recordIndex) => request(`/api/model-simulations/custom/${encodeURIComponent(sessionId)}/reveal`, {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ record_index: Number(recordIndex) }),
  }),
  revealComparison: (sessionId, recordIndex) => request(`/api/model-simulations/compare/${encodeURIComponent(sessionId)}/reveal`, {
    method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ record_index: Number(recordIndex) }),
  }),
  historyList: () => request('/api/history'),
  history: (code) => request(`/api/history/${encodeURIComponent(code)}`),
  scenario: (payload) => request('/api/scenario', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(payload) }),
  dataQuality: () => request('/api/data-quality', {}, 90000),
  ask: (query) => request('/api/assistant/query', { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ query }) }),
};
