import { api } from '../services/api.js';
import { horizontalBars } from '../components/charts.js';

const numeric = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};
const fixed = (value, digits = 2) => {
  const parsed = numeric(value);
  return parsed === null ? 'N/A' : parsed.toFixed(digits);
};
const metric = (value, digits = 2, suffix = '') => {
  const parsed = numeric(value);
  return parsed === null ? 'N/A' : `${parsed.toFixed(digits)}${suffix}`;
};
const ratioPercent = (value, digits = 1) => {
  const parsed = numeric(value);
  return parsed === null ? 'N/A' : `${(parsed * 100).toFixed(digits)}%`;
};
const escape = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const officialUrl = (value = '') => String(value).startsWith('https://paimana-proj.mospi.gov.in/') ? String(value) : '';
const shortFingerprint = (value) => value ? String(value).replace('sha256:', '').slice(0, 12) : 'Not recorded';

function yearOptions(years, selected) {
  return years.map((item) => `<option value="${item.year}" ${item.year === selected ? 'selected' : ''}>${item.year} · ${item.completed_projects} lifecycle projects</option>`).join('');
}

function predictionCard(prediction, actual = null) {
  const source = actual ? officialUrl(actual.source_url) : '';
  const range = prediction.expected_range;
  const confidenceText = prediction.confidence_calibration_status === 'not_calibrated_for_live_lifecycle_retrain'
    ? 'Not calibrated for this live retrain'
    : `${metric(prediction.model_confidence_percentage, 1, '%')} · ${escape(prediction.confidence_calibration_status || 'unavailable').replaceAll('_', ' ')}`;
  const factors = Array.isArray(prediction.shap_explanation) ? prediction.shap_explanation : [];
  const explanation = factors.length
    ? horizontalBars(factors.map((factor) => ({ label: `${factor.feature} (${factor.direction})`, value: Math.abs(factor.impact) })), { format: (v) => fixed(v, 3) })
    : '<p class="muted">No explanation factors were generated for this prediction.</p>';
  return `<div class="notice compact"><strong>Run identity:</strong> ${escape(prediction.run_id || 'Not recorded')} · <strong>dataset:</strong> ${escape(shortFingerprint(prediction.dataset_fingerprint))}</div>
  <div class="model-grid">
    <section class="panel">
      <span class="kicker">Monthly lifecycle AI prediction generated first</span>
      <h2>${escape(prediction.project.project_name)}</h2>
      <div class="detail-financial">
        <div><span>Predicted cost overrun</span><strong>${metric(prediction.predicted_cost_overrun, 2, '%')}</strong></div>
        <div><span>Predicted delay</span><strong>${metric(prediction.predicted_delay_days, 2, ' days')}</strong></div>
        <div><span>Predicted risk</span><strong>${escape(prediction.predicted_risk)} · ${metric(prediction.risk_probability_percentage, 1, '%')}</strong></div>
        <div><span>Prediction snapshot</span><strong>${escape(prediction.snapshot_date || 'Unavailable')}</strong></div>
        <div><span>Official history snapshots</span><strong>${prediction.history_snapshots ?? 'N/A'}</strong></div>
        <div><span>Confidence calibration</span><strong>${confidenceText}</strong></div>
        <div><span>Actual outcome sent yet?</span><strong>${prediction.audit.actual_outcomes_sent_to_browser ? 'Yes' : 'No'}</strong></div>
      </div>
      <h3>Lifecycle inputs visible to the model</h3>
      <div class="detail-financial">
        <div><span>Feature count</span><strong>${Object.keys(prediction.model_inputs || {}).length}</strong></div>
        <div><span>Approved cost</span><strong>${numeric(prediction.model_inputs.approved_cost_cr) === null ? 'N/A' : `₹${fixed(prediction.model_inputs.approved_cost_cr)} Cr`}</strong></div>
        <div><span>Revised cost</span><strong>${numeric(prediction.model_inputs.revised_cost_cr) === null ? 'N/A' : `₹${fixed(prediction.model_inputs.revised_cost_cr)} Cr`}</strong></div>
        <div><span>Expenditure ratio</span><strong>${ratioPercent(prediction.model_inputs.expenditure_ratio)}</strong></div>
        <div><span>Schedule slippage</span><strong>${metric(prediction.model_inputs.schedule_slippage_days, 2, ' days')}</strong></div>
        <div><span>Duration ratio</span><strong>${ratioPercent(prediction.model_inputs.duration_ratio)}</strong></div>
        <div><span>Sector</span><strong>${escape(prediction.model_inputs.sector ?? 'N/A')}</strong></div>
        <div><span>Implementing agency</span><strong>${escape(prediction.model_inputs.implementing_agency ?? 'N/A')}</strong></div>
      </div>
      ${range ? `<div class="notice compact"><strong>Uncertainty range:</strong> Cost P10–P90 ${metric(range.cost_overrun_percentage?.p10, 2, '%')} to ${metric(range.cost_overrun_percentage?.p90, 2, '%')}; delay P10–P90 ${metric(range.delay_days?.p10, 2, ' days')} to ${metric(range.delay_days?.p90, 2, ' days')}.</div>` : ''}
    </section>
    <section class="panel">
      <span class="kicker">Explainability</span>
      <h2>Why the lifecycle model predicted this</h2>
      ${explanation}
      <div class="notice compact"><strong>Leakage audit:</strong> This project is excluded from the selected training years: ${prediction.audit.project_excluded_from_training ? 'YES' : 'NO'}.</div>
    </section>
  </div>
  ${actual ? `<section class="panel"><span class="kicker">Official outcome revealed after prediction</span><h2>Prediction vs actual</h2><div class="detail-financial"><div><span>AI cost overrun</span><strong>${metric(prediction.predicted_cost_overrun, 2, '%')}</strong></div><div><span>Actual cost overrun</span><strong>${metric(actual.actual_cost_overrun, 2, '%')}</strong></div><div><span>Absolute cost error</span><strong>${metric(actual.cost_error_absolute_pp, 2, ' pp')}</strong></div><div><span>AI delay</span><strong>${metric(prediction.predicted_delay_days, 2, ' days')}</strong></div><div><span>Actual delay</span><strong>${metric(actual.actual_delay_days, 2, ' days')}</strong></div><div><span>Absolute delay error</span><strong>${metric(actual.delay_error_absolute_days, 2, ' days')}</strong></div><div><span>AI / actual risk</span><strong>${escape(prediction.predicted_risk)} / ${escape(actual.actual_risk)}</strong></div><div><span>Recorded completion</span><strong>${escape(actual.completion_date ?? 'N/A')}</strong></div></div><div class="notice compact"><strong>Reveal audit:</strong> ${escape(actual.reveal_policy)} · run ${escape(actual.run_id || 'Not recorded')}</div>${source ? `<a class="secondary-btn" href="${escape(source)}" target="_blank" rel="noopener noreferrer">Open official PAIMANA source</a>` : ''}</section>` : '<div class="notice compact"><strong>Actual outcome is still hidden.</strong> Click Reveal Actual Outcome only after the judge has seen the AI prediction.</div>'}`;
}

function trainingReceipt(registryRun, session = null, restored = false) {
  if (!registryRun) return 'No lifecycle model has been trained in this browser session yet.';
  const quality = registryRun.metrics?.metadata?.feature_quality || {};
  const baseline = registryRun.baseline_comparison || {};
  const algorithms = registryRun.selected_algorithms || {};
  const riskF1 = numeric(registryRun.metrics?.risk_model?.macro_f1);
  const baselineRiskF1 = numeric(baseline.risk_macro_f1);
  const sessionAudit = session ? ` <strong>Leakage guard:</strong> ${escape(session.leakage_guard)} Browser received actual held-out outcomes: <strong>${session.actual_outcomes_sent_to_browser ? 'YES' : 'NO'}</strong>.` : '';
  return `${restored ? '<strong>Restored active browser-session run.</strong> ' : ''}<strong>${escape(registryRun.model_version)} retrained from scratch.</strong> <strong>Run ID:</strong> ${escape(registryRun.run_id || 'Not recorded')} · <strong>dataset fingerprint:</strong> ${escape(shortFingerprint(registryRun.dataset_fingerprint))}. Training: ${escape(registryRun.training_years)} · internal validation: ${escape(registryRun.internal_validation_year)} · untouched future holdout: ${escape(registryRun.testing_years)}. <strong>Lifecycle features:</strong> ${registryRun.feature_count ?? 'N/A'} retained · ${fixed(quality.removed_invalid_feature_count, 0)} rejected by the selected window audit. <strong>Selected models:</strong> cost ${escape(algorithms.cost || 'unknown')} · delay ${escape(algorithms.delay || 'unknown')} · risk Random Forest. <strong>Fresh holdout metrics:</strong> cost MAE ${metric(registryRun.metrics?.cost_model?.MAE, 2, ' pp')} · delay MAE ${metric(registryRun.metrics?.delay_model?.MAE, 2, ' days')} · risk macro-F1 ${riskF1 === null ? 'N/A' : `${(riskF1 * 100).toFixed(1)}%`}. <strong>Five-feature benchmark:</strong> cost MAE ${metric(baseline.cost_mae, 2, ' pp')} · delay MAE ${metric(baseline.delay_mae, 2, ' days')} · risk macro-F1 ${baselineRiskF1 === null ? 'N/A' : `${(baselineRiskF1 * 100).toFixed(1)}%`}. <strong>Feature quality:</strong> ${metric(quality.data_quality_score, 1, '%')}.${sessionAudit}<br><a class="secondary-btn" href="#/prediction-accuracy">View Prediction Accuracy</a>`;
}

export async function ModelSimulationPage(root) {
  const catalog = await api.simulationVersions();
  if (!catalog.lifecycle_data_available) {
    root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1><p>Live retraining requires the official processed PAIMANA monthly lifecycle dataset.</p></div></header>
    <section class="panel"><div class="error-state">Official PAIMANA monthly lifecycle dataset is not available in this checkout.</div><p class="muted">Refresh the official archive separately, then rebuild <code>data/processed/paimana_monthly_snapshots.csv</code>. The Retrain button does not download or parse PAIMANA PDFs.</p><button class="primary-btn" disabled>Retrain Lifecycle Models Live</button></section>`;
    return;
  }
  const years = catalog.data_years || [];
  if (!years.length) throw new Error('No identity-verified PAIMANA lifecycle years are available.');

  const savedRun = api.getActiveLifecycleRun();
  const yearNumbers = years.map((item) => item.year);
  const savedStart = Number(savedRun?.start_year);
  const savedEnd = Number(savedRun?.end_year);
  const defaultStart = yearNumbers.includes(savedStart) ? savedStart : yearNumbers[0];
  const preferredEnd = yearNumbers.filter((year) => year <= 2015).at(-1);
  const fallbackEnd = preferredEnd || yearNumbers[Math.max(0, Math.floor(yearNumbers.length / 2) - 1)];
  const defaultEnd = yearNumbers.includes(savedEnd) ? savedEnd : fallbackEnd;

  root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1><p>Choose a historical training range and retrain the monthly lifecycle cost, delay, and risk models from scratch. The active run is kept for this browser session when you navigate to other pages.</p></div></header>
  <div class="notice"><strong>Leakage rule:</strong> Algorithm selection happens inside the selected training period. Projects completed after the training cutoff are held out from fitting and are used only for future evaluation and judge-selected prediction.</div>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 1</span><h2>Choose training years and retrain lifecycle models</h2></div></div>
    <div class="filters">
      <label>Training start year<select id="custom-start">${yearOptions(years, defaultStart)}</select></label>
      <label>Training end year<select id="custom-end">${yearOptions(years, defaultEnd)}</select></label>
      <button class="primary-btn" id="custom-train">Retrain Lifecycle Models Live</button>
    </div>
    <div id="training-receipt" class="notice compact">No lifecycle model has been trained in this browser session yet.</div>
  </section>
  <section class="panel">
    <div class="panel-head"><div><span class="kicker">Step 2</span><h2>Judge chooses an unseen future project</h2></div></div>
    <div class="filters">
      <label>Held-out completion year<select id="custom-test-year" disabled><option>Retrain first</option></select></label>
      <label>Official held-out project<select id="custom-project" disabled><option>Select a test year first</option></select></label>
      <button class="secondary-btn" id="random-project" disabled>Pick Random Unseen Project</button>
      <button class="primary-btn" id="custom-predict" disabled>Generate Lifecycle Prediction</button>
      <button class="secondary-btn" id="custom-reveal" disabled>Reveal Actual Outcome</button>
    </div>
    <div id="held-out-note" class="notice compact">After retraining, only projects completed after the selected training end year will be offered here.</div>
  </section>
  <div id="custom-output"></div>`;

  const start = root.querySelector('#custom-start');
  const end = root.querySelector('#custom-end');
  const trainButton = root.querySelector('#custom-train');
  const receipt = root.querySelector('#training-receipt');
  const testYear = root.querySelector('#custom-test-year');
  const project = root.querySelector('#custom-project');
  const randomButton = root.querySelector('#random-project');
  const predictButton = root.querySelector('#custom-predict');
  const revealButton = root.querySelector('#custom-reveal');
  const heldOutNote = root.querySelector('#held-out-note');
  const output = root.querySelector('#custom-output');

  let session = null;
  let projectRows = [];
  let prediction = null;
  let actual = null;

  const resetPrediction = () => {
    prediction = null;
    actual = null;
    revealButton.disabled = true;
    output.innerHTML = '';
  };

  const loadProjects = async () => {
    if (!session || !testYear.value) return;
    resetPrediction();
    project.disabled = true;
    predictButton.disabled = true;
    randomButton.disabled = true;
    heldOutNote.innerHTML = '<div class="loading">Loading held-out official lifecycle projects…</div>';
    try {
      const response = await api.customSimulationProjects(session.session_id, Number(testYear.value));
      if (session.run_id && response.run_id !== session.run_id) throw new Error('Held-out project response belongs to a different model run. Retrain this range.');
      projectRows = response.items;
      project.innerHTML = projectRows.map((row) => `<option value="${row.record_index}">${escape(row.project_id)} · ${escape(row.project_name)}</option>`).join('');
      project.disabled = !projectRows.length;
      predictButton.disabled = !projectRows.length;
      randomButton.disabled = !projectRows.length;
      heldOutNote.innerHTML = `<strong>${projectRows.length} held-out projects available for ${escape(response.year)}.</strong> ${escape(response.note)} No actual cost, delay, completion date, or final expenditure has been sent to this page.`;
    } catch (error) {
      projectRows = [];
      project.innerHTML = '<option>No projects available</option>';
      heldOutNote.innerHTML = `<div class="error-state">${escape(error.message)}</div><p class="muted">If the backend was restarted or this run was replaced, retrain this saved range to create a new judge session.</p>`;
    }
  };

  const generatePrediction = async () => {
    if (!session || project.disabled) return;
    resetPrediction();
    predictButton.disabled = true;
    output.innerHTML = '<div class="loading">Generating prediction from the freshly trained monthly lifecycle model…</div>';
    try {
      prediction = await api.predictCustomSimulation(session.session_id, Number(project.value));
      if (session.run_id && prediction.run_id !== session.run_id) throw new Error('Prediction belongs to a different model run. Retrain this range.');
      revealButton.disabled = false;
      output.innerHTML = predictionCard(prediction);
    } catch (error) {
      output.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
    } finally {
      predictButton.disabled = false;
    }
  };

  trainButton.addEventListener('click', async () => {
    resetPrediction();
    const startYear = Number(start.value);
    const endYear = Number(end.value);
    if (startYear > endYear) {
      receipt.innerHTML = '<div class="error-state">Training start year cannot be after training end year.</div>';
      return;
    }
    trainButton.disabled = true;
    testYear.disabled = true;
    project.disabled = true;
    predictButton.disabled = true;
    randomButton.disabled = true;
    receipt.innerHTML = '<div class="loading">Building selected PAIMANA lifecycle cohort → auditing features → selecting cost/delay regressors on internal temporal validation → fitting final cost, delay, and risk models → evaluating future holdout…</div>';
    try {
      const registryRun = await api.retrainModel(startYear, endYear);
      const activeBase = {
        window: registryRun.window,
        model_version: registryRun.model_version,
        run_id: registryRun.run_id,
        dataset_fingerprint: registryRun.dataset_fingerprint,
        start_year: startYear,
        end_year: endYear,
        receipt: registryRun,
      };
      api.setActiveLifecycleRun(activeBase);
      receipt.innerHTML = `${trainingReceipt(registryRun)}<div class="loading">Preparing judge-controlled held-out project session…</div>`;

      try {
        session = await api.trainCustomSimulation(startYear, endYear, registryRun.run_id);
        if (registryRun.run_id && session.run_id !== registryRun.run_id) {
          throw new Error('Judge session did not bind to the exact retrained model run. Retrain again.');
        }
        if (registryRun.dataset_fingerprint && session.dataset_fingerprint !== registryRun.dataset_fingerprint) {
          throw new Error('Judge session dataset fingerprint does not match the retrained model. Retrain again.');
        }
        api.setActiveLifecycleRun({ ...activeBase, session });
      } catch (sessionError) {
        session = null;
        receipt.innerHTML = `${trainingReceipt(registryRun)}<div class="error-state">The model was trained and saved, but the judge session could not be created: ${escape(sessionError.message)}</div>`;
        return;
      }

      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} held-out projects</option>`).join('');
      testYear.disabled = !eligible.length;
      receipt.innerHTML = trainingReceipt(registryRun, session);
      if (eligible.length) await loadProjects();
    } catch (error) {
      session = null;
      receipt.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
    } finally {
      trainButton.disabled = false;
    }
  });

  testYear.addEventListener('change', loadProjects);
  project.addEventListener('change', resetPrediction);
  predictButton.addEventListener('click', generatePrediction);
  randomButton.addEventListener('click', async () => {
    if (!projectRows.length) return;
    const chosen = projectRows[Math.floor(Math.random() * projectRows.length)];
    project.value = String(chosen.record_index);
    await generatePrediction();
  });
  revealButton.addEventListener('click', async () => {
    if (!session || !prediction) return;
    revealButton.disabled = true;
    try {
      actual = await api.revealCustomSimulation(session.session_id, prediction.record_index);
      if (session.run_id && actual.run_id !== session.run_id) throw new Error('Reveal response belongs to a different model run.');
      output.innerHTML = predictionCard(prediction, actual);
    } catch (error) {
      output.innerHTML += `<div class="error-state">${escape(error.message)}</div>`;
      revealButton.disabled = false;
    }
  });

  if (savedRun?.receipt && savedRun.start_year === defaultStart && savedRun.end_year === defaultEnd) {
    receipt.innerHTML = trainingReceipt(savedRun.receipt, savedRun.session || null, true);
    if (savedRun.session?.session_id) {
      session = savedRun.session;
      const eligible = session.eligible_test_years || [];
      testYear.innerHTML = eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} held-out projects</option>`).join('');
      testYear.disabled = !eligible.length;
      if (eligible.length) await loadProjects();
    } else {
      heldOutNote.innerHTML = '<strong>The trained model is still the active browser-session run.</strong> Retrain only if you need a new judge session or a different year range.';
    }
  }
}