import { api } from '../services/api.js';
import { horizontalBars, lineChart } from '../components/charts.js';

const value = (number, digits = 2) => {
  if (number === null || number === undefined || number === '' || Number.isNaN(Number(number))) return 'N/A';
  return Number(number).toFixed(digits);
};
const formatMetric = (number, digits = 2, suffix = '') => `${value(number, digits)}${value(number, digits) === 'N/A' ? '' : suffix}`;
const escape = (text = '') => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const shortFingerprint = (fingerprint) => fingerprint ? String(fingerprint).replace('sha256:', '').slice(0, 12) : 'Not recorded';

function errorBuckets(rows, key) {
  return [[0,5],[5,15],[15,30],[30,Infinity]].map(([from,to]) => ({
    label: `${from}-${to === Infinity ? '∞' : to}`,
    value: rows.filter(r => { const e=Math.abs(Number(r[key])); return e>=from && e<to; }).length,
  }));
}

function runOptions(runs, selected) {
  return runs.map((run) => {
    const label = `${run.training_start}–${run.training_end}${run.complete ? '' : ' · summary only'}`;
    return `<option value="${escape(run.window)}" ${run.window === selected ? 'selected' : ''}>${escape(label)}</option>`;
  }).join('');
}

export async function PredictionAccuracyPage(root) {
  const activeRun = api.getActiveLifecycleRun();
  let selected = activeRun?.window || api.getValidationModel();
  let runs = [];

  const selectorPanel = () => `<section class="panel"><div class="panel-head"><div><span class="kicker">Model selection</span><h2>Available lifecycle runs</h2></div><div class="filters compact-filters"><select id="validation-model" class="data-input">${runOptions(runs, selected)}</select><button class="secondary-btn" id="refresh-validation-runs">Refresh</button><a class="secondary-btn" href="#/model-simulation">Train another range</a></div></div><p class="muted">Only year ranges that actually have lifecycle evaluation artifacts are listed. A freshly retrained range appears here immediately.</p></section>`;

  const bindSelector = () => {
    root.querySelector('#validation-model')?.addEventListener('change', (event) => {
      selected = event.target.value;
      api.setValidationModel(selected);
      render();
    });
    root.querySelector('#refresh-validation-runs')?.addEventListener('click', render);
  };

  async function render() {
    root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1><p>Loading available lifecycle runs and validation evidence…</p></div></header><div class="loading">Loading prediction accuracy…</div>`;

    let registry;
    try {
      registry = await api.lifecycleRuns();
    } catch (error) {
      root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1></div></header><section class="panel"><div class="error-state"><strong>Unable to load lifecycle model registry</strong><span>${escape(error.message)}</span><button class="secondary-btn" id="retry-registry">Retry</button></div></section>`;
      root.querySelector('#retry-registry')?.addEventListener('click', render);
      return;
    }

    runs = (registry.items || []).filter((item) => item.summary_available && !item.in_progress);
    if (!runs.length) {
      root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1></div></header><section class="panel"><div class="error-state"><strong>No lifecycle model runs are available yet.</strong><span>Train a year range in Model Simulation first.</span><a class="primary-btn" href="#/model-simulation">Open Model Simulation</a></div></section>`;
      return;
    }

    if (!selected) {
      selected = (runs.find((item) => item.window === '2001_2015') || runs[0]).window;
      api.setValidationModel(selected);
    }

    const selectedRun = runs.find((item) => item.window === selected);
    if (!selectedRun) {
      root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1></div></header>${selectorPanel()}<section class="panel"><div class="error-state"><strong>Lifecycle model ${escape(selected)} has not been trained in this runtime.</strong><span>Retrain that exact range in Model Simulation. Prediction Accuracy no longer substitutes unrelated legacy metrics.</span><a class="primary-btn" href="#/model-simulation">Train this range</a></div></section>`;
      bindSelector();
      return;
    }

    const [reportResult, validationResult, rollingResult] = await Promise.allSettled([
      api.validationReport(selected),
      api.predictionValidation(100, selected),
      api.rollingValidation(selected),
    ]);

    if (reportResult.status !== 'fulfilled') {
      root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest</span><h1>Prediction Accuracy Dashboard</h1></div></header>${selectorPanel()}<section class="panel"><div class="error-state"><strong>Unable to load lifecycle evaluation summary</strong><span>${escape(reportResult.reason?.message || 'Unknown error')}</span><button class="secondary-btn" id="retry-report">Retry</button></div></section>`;
      bindSelector();
      root.querySelector('#retry-report')?.addEventListener('click', render);
      return;
    }

    const report = reportResult.value;
    const validation = validationResult.status === 'fulfilled' ? validationResult.value : { items: [], total: 0 };
    const rolling = rollingResult.status === 'fulfilled' ? rollingResult.value : { folds: [], fold_count: 0, status: 'not_generated' };
    const rows = validation.items || [];
    const lifecycle = report.model_family === 'monthly_lifecycle';
    const risk = report.risk_model || report.risk_classification || {};
    const delayMae = report.delay_model?.MAE_days ?? report.delay_model?.MAE;
    const delayRmse = report.delay_model?.RMSE_days ?? report.delay_model?.RMSE;
    const confidenceValues = rows
      .map(row => row.model_confidence_percentage)
      .filter(item => item !== null && item !== undefined && item !== '')
      .map(Number)
      .filter(Number.isFinite);
    const averageConfidence = confidenceValues.length ? confidenceValues.reduce((total, item) => total + item, 0) / confidenceValues.length : null;
    const metadata = report.metadata || {};
    const quality = metadata.feature_quality || {};
    const sectors = Object.entries(report.sector_validation?.sectors || {});
    const shapValidation = metadata.shap_validation || {};
    const shapTargets = lifecycle ? Object.entries(report.shap || {}) : Object.entries(shapValidation.targets || {});
    const runId = metadata.run_id || selectedRun.run_id || null;
    const datasetFingerprint = metadata.dataset_fingerprint || selectedRun.dataset_fingerprint || null;
    const validationWarning = validationResult.status === 'rejected'
      ? `<div class="notice compact"><strong>Project-wise validation rows unavailable:</strong> ${escape(validationResult.reason?.message || 'Not generated for this run')}. Summary metrics are still valid.</div>`
      : '';
    const rollingWarning = rollingResult.status === 'rejected'
      ? `<div class="notice compact"><strong>Rolling validation unavailable:</strong> ${escape(rollingResult.reason?.message || 'Not generated for this run')}.</div>`
      : '';

    root.innerHTML = `<header class="page-head"><div><span class="kicker">Historical cutoff backtest · ${escape(report.model_version || selected)}</span><h1>Prediction Accuracy Dashboard</h1><p>Metrics are loaded from the exact selected lifecycle run; later completed projects remain the out-of-time holdout.</p></div></header>
      ${selectorPanel()}
      <div class="notice"><strong>Model family:</strong> ${lifecycle ? 'Monthly Lifecycle' : 'Legacy verification'} · <strong>Training:</strong> ${metadata.training_start ?? 'N/A'}-${metadata.training_end ?? 'N/A'} · <strong>Future holdout:</strong> ${metadata.evaluated_test_start ?? 'N/A'}-${metadata.evaluated_test_end ?? 'N/A'}. ${metadata.leakage_policy || metadata.future_information_policy || ''}</div>
      <div class="notice compact"><strong>Run ID:</strong> ${escape(runId || 'Not recorded for this persisted artifact')} · <strong>Dataset fingerprint:</strong> ${escape(shortFingerprint(datasetFingerprint))} · <strong>Generated:</strong> ${escape(metadata.created_at || selectedRun.created_at || 'Not recorded')}</div>
      <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Cost MAE</span><strong class="stat-value">${formatMetric(report.cost_model?.MAE)} pp</strong><small class="stat-note">RMSE ${formatMetric(report.cost_model?.RMSE)} · MAPE ${formatMetric(report.cost_model?.MAPE)}% · R² ${value(report.cost_model?.R2, 3)}</small></article><article class="stat-card amber"><span class="stat-eyebrow">Delay MAE</span><strong class="stat-value">${formatMetric(delayMae)} days</strong><small class="stat-note">RMSE ${formatMetric(delayRmse)} · ${lifecycle ? `MAPE ${formatMetric(report.delay_model?.MAPE)}%` : `log-target RMSE ${value(report.delay_model?.log_target_RMSE, 4)}`} · R² ${value(report.delay_model?.R2, 3)}</small></article><article class="stat-card green"><span class="stat-eyebrow">${lifecycle ? 'Risk macro-F1' : 'Risk classification'}</span><strong class="stat-value">${formatMetric((lifecycle ? risk.macro_f1 : risk.accuracy) * 100, 1, '%')}</strong><small class="stat-note">${lifecycle ? `Accuracy ${formatMetric(risk.accuracy * 100, 1, '%')} · macro precision ${formatMetric(risk.macro_precision * 100, 1, '%')} · macro recall ${formatMetric(risk.macro_recall * 100, 1, '%')}` : `Precision ${formatMetric(risk.precision * 100, 1, '%')} · recall ${formatMetric(risk.recall * 100, 1, '%')} · F1 ${formatMetric(risk.f1 * 100, 1, '%')}`}</small></article><article class="stat-card"><span class="stat-eyebrow">Model confidence</span><strong class="stat-value">${formatMetric(averageConfidence, 1, '%')}</strong><small class="stat-note">${lifecycle ? 'Not generated for this lifecycle run' : 'Earlier-year calibrated interval coverage'}</small></article></div>
      <div class="stat-grid"><article class="stat-card blue"><span class="stat-eyebrow">Features used</span><strong class="stat-value">${value(metadata.feature_count, 0)}</strong><small class="stat-note">Only audited official-data features</small></article><article class="stat-card amber"><span class="stat-eyebrow">Removed invalid</span><strong class="stat-value">${value(quality.removed_invalid_feature_count, 0)}</strong><small class="stat-note">Empty, constant, synthetic, or unavailable fields</small></article><article class="stat-card green"><span class="stat-eyebrow">Data quality</span><strong class="stat-value">${formatMetric(quality.data_quality_score, 1, '%')}</strong><small class="stat-note">Availability across retained features</small></article><article class="stat-card"><span class="stat-eyebrow">Run status</span><strong class="stat-value">${escape(selectedRun.status.replaceAll('_', ' '))}</strong><small class="stat-note">Validation rows: ${selectedRun.has_validation_rows ? 'available' : 'not generated in this checkout'}</small></article></div>
      ${validationWarning}${rollingWarning}
      <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">${sectors.length ? 'Validation-safe sector breakdown' : 'Optional artifact'}</span><h2>Sector performance</h2></div></div><div class="table-wrap">${sectors.length ? `<table class="data-table compact-table"><thead><tr><th>Sector</th><th>Projects</th><th>Cost MAE</th><th>Delay MAE</th></tr></thead><tbody>${sectors.map(([name, item]) => `<tr><td>${escape(name)}</td><td>${item.projects}</td><td>${formatMetric(item.cost_mae)} pp</td><td>${formatMetric(item.delay_mae)} days</td></tr>`).join('')}</tbody></table>` : '<p class="muted">Not generated for this run.</p>'}</div></section><section class="panel"><span class="kicker">${lifecycle ? 'Lifecycle feature importance' : 'Validated explanations'}</span><h2>${lifecycle ? 'SHAP / feature importance' : 'SHAP quality'}</h2>${lifecycle ? shapTargets.map(([target, item]) => `<p><strong>${escape(target)}:</strong> ${escape((item.features || []).slice(0, 5).map(feature => feature.feature).join(', ') || 'Not generated for this run')}</p>`).join('') || '<p class="muted">Not generated for this run.</p>' : `<div class="notice compact"><strong>${shapValidation.validated ? 'Validated' : 'Review warning'}:</strong> explanation factors are checked against the retained feature contract.</div>${shapTargets.map(([target, item]) => `<p><strong>${escape(target)}:</strong> ${escape((item.meaningful_expected_factors || []).join(', ') || item.status)}</p>`).join('')}`}</section></div>
      <section class="panel"><div class="panel-head"><div><span class="kicker">Expanding-window temporal validation · ${value(rolling.fold_count, 0)} folds</span><h2>Reliability across unseen completion years</h2></div></div>${rolling.folds?.length ? `<div class="model-grid"><div><h3>Cost MAE by test year</h3>${lineChart(rolling.folds.map((fold) => ({ label: fold.test_year, value: fold.cost_MAE })), { suffix: ' pp' })}</div><div><h3>Delay MAE by test year</h3>${lineChart(rolling.folds.map((fold) => ({ label: fold.test_year, value: fold.delay_MAE_days })), { suffix: ' days' })}</div></div><div class="notice compact"><strong>Average risk macro F1:</strong> ${formatMetric(rolling.average_risk_f1 * 100, 1, '%')}. Every fold trains only on years before its displayed test year.</div>` : '<div class="notice compact">Rolling validation not generated for this lifecycle run.</div>'}</section>
      ${rows.length ? `<div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost escalation</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_cost_overrun })), { suffix: '%' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_cost_overrun })), { suffix: '%' })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Schedule extension</span><h2>Predicted vs actual</h2></div></div><h3>AI predicted</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.predicted_delay_days })), { suffix: ' days' })}<h3>Actual final outcome</h3>${lineChart(rows.map((row) => ({ label: row.project_id, value: row.actual_delay_days })), { suffix: ' days' })}</section></div>
      <div class="model-grid"><section class="panel"><div class="panel-head"><div><span class="kicker">Cost error</span><h2>Error distribution (percentage points)</h2></div></div>${horizontalBars(errorBuckets(rows, 'cost_error'), { format: (count) => `${count} projects` })}</section><section class="panel"><div class="panel-head"><div><span class="kicker">Delay error</span><h2>Error distribution (days)</h2></div></div>${horizontalBars(errorBuckets(rows, 'delay_error'), { format: (count) => `${count} projects` })}</section></div>
      <section class="panel"><div class="panel-head"><div><span class="kicker">${validation.total} completed projects</span><h2>Project-wise forecast accuracy</h2></div><div class="filters compact-filters"><label>Error metric<select id="validation-sort-metric"><option value="cost_error">Cost error</option><option value="delay_error">Delay error</option></select></label><button class="secondary-btn" id="validation-sort-low">Lowest error first</button><button class="secondary-btn" id="validation-sort-high">Highest error first</button></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Project ID</th><th>Project name</th><th>Predicted cost</th><th>Cost P10–P90</th><th>Actual cost</th><th>Cost error</th><th>Predicted delay</th><th>Delay P10–P90</th><th>Actual delay</th><th>Delay error</th></tr></thead><tbody id="validation-rows"></tbody></table></div></section>` : '<section class="panel"><div class="notice"><strong>Project-wise validation rows were not generated or are not stored in this checkout.</strong> The model summary above remains available and no unrelated validation rows are substituted.</div></section>'}`;

    bindSelector();
    if (!rows.length) return;

    const tableBody = root.querySelector('#validation-rows');
    const sortMetricSelect = root.querySelector('#validation-sort-metric');
    const renderRows = (sortKey = 'cost_error', direction = 'asc') => {
      const sorted = [...rows].sort((a, b) => {
        const leftRaw = a[sortKey];
        const rightRaw = b[sortKey];
        const left = leftRaw === null || leftRaw === undefined || leftRaw === '' || !Number.isFinite(Number(leftRaw)) ? Infinity : Math.abs(Number(leftRaw));
        const right = rightRaw === null || rightRaw === undefined || rightRaw === '' || !Number.isFinite(Number(rightRaw)) ? Infinity : Math.abs(Number(rightRaw));
        return direction === 'asc' ? left - right : right - left;
      });
      tableBody.innerHTML = sorted.map((row) => `<tr><td>${escape(row.project_id || 'Not published')}</td><td>${escape(row.project_name || 'Not reported')}</td><td>${formatMetric(row.predicted_cost_overrun, 2, '%')}</td><td>${row.predicted_cost_p10 == null || row.predicted_cost_p90 == null ? 'N/A' : `${formatMetric(row.predicted_cost_p10, 2, '%')}–${formatMetric(row.predicted_cost_p90, 2, '%')}`}</td><td>${formatMetric(row.actual_cost_overrun, 2, '%')}</td><td>${formatMetric(row.cost_error)} pp</td><td>${formatMetric(row.predicted_delay_days, 2, ' days')}</td><td>${row.predicted_delay_p10 == null || row.predicted_delay_p90 == null ? 'N/A' : `${value(row.predicted_delay_p10)}–${value(row.predicted_delay_p90)} days`}</td><td>${formatMetric(row.actual_delay_days, 2, ' days')}</td><td>${formatMetric(row.delay_error, 2, ' days')}</td></tr>`).join('');
    };
    renderRows();
    root.querySelector('#validation-sort-low')?.addEventListener('click', () => renderRows(sortMetricSelect.value, 'asc'));
    root.querySelector('#validation-sort-high')?.addEventListener('click', () => renderRows(sortMetricSelect.value, 'desc'));
  }

  await render();
}