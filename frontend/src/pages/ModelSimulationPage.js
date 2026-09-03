import { api } from '../services/api.js';

const missing = (value) => value === null || value === undefined || value === '' || (typeof value === 'number' && !Number.isFinite(value));
const fixed = (value, digits = 2) => missing(value) ? 'N/A' : Number(value).toFixed(digits);
const escape = (value = '') => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const shortFingerprint = (value) => value ? String(value).replace('sha256:', '').slice(0, 12) : 'Not recorded';

function yearOptions(years, selected) {
  return years.map((item) => `<option value="${item.year}" ${item.year === selected ? 'selected' : ''}>${item.year} · ${item.completed_projects} lifecycle projects</option>`).join('');
}

function improvementLabel(value) {
  if (missing(value)) return 'N/A';
  const number = Number(value);
  if (number > 0) return `${fixed(number)}% better`;
  if (number < 0) return `${fixed(Math.abs(number))}% worse`;
  return '0.00% change';
}

function overallCard(overall, experiment) {
  if (!overall || !experiment) return '';
  const pairedCost = overall.paired_project_cost_comparison || overall.paired_project_comparison || {};
  const pairedDelay = overall.paired_project_delay_comparison || {};
  const costCi = pairedCost.improvement_95pct_ci || [];
  const delayCi = pairedDelay.improvement_95pct_ci || [];
  const stage = overall.stage_balanced || {};
  const costOnly = experiment.scope === 'cost' || overall.delay_policy === 'production_retained';
  const delaySummary = costOnly
    ? `<div><span>Delay model</span><strong>Production retained</strong></div>
       <div><span>Production delay MAE</span><strong>${fixed(overall.production_delay_mae)} days</strong></div>`
    : `<div><span>Production delay MAE</span><strong>${fixed(overall.production_delay_mae)} days</strong></div>
       <div><span>Challenger delay MAE</span><strong>${fixed(overall.experiment_delay_mae)} days</strong></div>
       <div><span>Delay improvement</span><strong>${escape(improvementLabel(overall.delay_improvement_percentage))}</strong></div>
       <div><span>Delay MAE reduction</span><strong>${fixed(overall.absolute_delay_mae_improvement_days)} days</strong></div>`;
  const delayEvidence = costOnly
    ? `<div><span>Delay experiment status</span><strong>Rejected; production unchanged</strong></div>`
    : `<div><span>Delay bootstrap chance better</span><strong>${missing(pairedDelay.probability_candidate_better) ? 'N/A' : `${fixed(Number(pairedDelay.probability_candidate_better) * 100, 1)}%`}</strong></div>
       <div><span>Delay improvement 95% CI</span><strong>${delayCi.length === 2 ? `${fixed(delayCi[0])}% to ${fixed(delayCi[1])}%` : 'N/A'}</strong></div>`;
  const delayStage = costOnly
    ? ''
    : `<div><span>Stage-balanced production delay MAE</span><strong>${fixed(stage.production_delay_mae)} days</strong></div>
       <div><span>Stage-balanced challenger delay MAE</span><strong>${fixed(stage.experiment_delay_mae)} days</strong></div>`;
  return `<section class="panel">
    <div class="panel-head"><div><span class="kicker">Fresh same-cohort comparison</span><h2>Production lifecycle vs ${escape(experiment.experiment_name || experiment.experiment_id)}</h2></div></div>
    <div class="detail-financial">
      <div><span>Production cost MAE</span><strong>${fixed(overall.production_cost_mae)} pp</strong></div>
      <div><span>Challenger cost MAE</span><strong>${fixed(overall.experiment_cost_mae)} pp</strong></div>
      <div><span>Cost improvement</span><strong>${escape(improvementLabel(overall.improvement_percentage))}</strong></div>
      <div><span>Cost MAE reduction</span><strong>${fixed(overall.absolute_mae_improvement_pp)} pp</strong></div>
      ${delaySummary}
      <div><span>Comparable test projects</span><strong>${missing(overall.comparison_test_projects) ? 'N/A' : overall.comparison_test_projects}</strong></div>
      <div><span>Comparable test snapshots</span><strong>${missing(overall.comparison_test_snapshots) ? 'N/A' : overall.comparison_test_snapshots}</strong></div>
      <div><span>Cost bootstrap chance better</span><strong>${missing(pairedCost.probability_candidate_better) ? 'N/A' : `${fixed(Number(pairedCost.probability_candidate_better) * 100, 1)}%`}</strong></div>
      ${delayEvidence}
      <div><span>Cost improvement 95% CI</span><strong>${costCi.length === 2 ? `${fixed(costCi[0])}% to ${fixed(costCi[1])}%` : 'N/A'}</strong></div>
      <div><span>Stage-balanced production cost MAE</span><strong>${fixed(stage.production_cost_mae)} pp</strong></div>
      <div><span>Stage-balanced challenger cost MAE</span><strong>${fixed(stage.experiment_cost_mae)} pp</strong></div>
      ${delayStage}
    </div>
    <div class="notice compact"><strong>Isolation:</strong> the challenger remains an experiment and is never auto-promoted to production.${costOnly ? ' Experiment 12 changes cost only; delay stays on the production model.' : ''}</div>
  </section>`;
}

function predictionCard(prediction, actual = null) {
  const comparison = prediction.comparison || {};
  const challenger = comparison.experiment || {};
  const reveal = actual?.comparison || null;
  const costOnly = challenger.scope === 'cost' || challenger.delay_policy === 'production_retained';
  const optionalResidual = !missing(challenger.predicted_remaining_cost_overrun)
    ? `<div><span>Predicted remaining overrun</span><strong>${fixed(challenger.predicted_remaining_cost_overrun)}%</strong></div>`
    : '';
  const optionalAnchor = !missing(challenger.current_observed_cost_escalation)
    ? `<div><span>Current observed escalation</span><strong>${fixed(challenger.current_observed_cost_escalation)}%</strong></div>`
    : '';
  const trajectoryCoverage = !missing(challenger.trajectory_features_available)
    ? `<div><span>Trajectory features available</span><strong>${challenger.trajectory_features_available}/${challenger.trajectory_feature_count || 'N/A'}</strong></div>`
    : '';
  const challengerDelay = !missing(challenger.predicted_delay_days)
    ? `<div><span>Challenger predicted delay</span><strong>${fixed(challenger.predicted_delay_days)} days</strong></div>
       <div><span>Delay prediction difference</span><strong>${fixed(comparison.delay_prediction_difference_days)} days</strong></div>`
    : '';
  const delayPolicy = costOnly
    ? `<div><span>Delay policy</span><strong>Production model retained</strong></div>`
    : challengerDelay;
  const revealDelay = costOnly
    ? `<div><span>Actual final delay</span><strong>${fixed(actual?.actual_delay_days)} days</strong></div>
       <div><span>Production delay error</span><strong>${fixed(actual?.delay_error_absolute_days)} days</strong></div>`
    : `<div><span>Actual final delay</span><strong>${fixed(actual?.actual_delay_days)} days</strong></div>
       <div><span>Production delay error</span><strong>${fixed(reveal?.production_delay_error_absolute_days)} days</strong></div>
       <div><span>Challenger delay error</span><strong>${fixed(reveal?.experiment_delay_error_absolute_days)} days</strong></div>
       <div><span>Challenger delay improvement</span><strong>${escape(improvementLabel(reveal?.individual_delay_error_improvement_percentage))}</strong></div>`;

  return `<section class="panel">
    <span class="kicker">Same held-out project · both predictions before reveal</span>
    <h2>${escape(prediction.project?.project_name || prediction.project?.project_id || 'Held-out project')}</h2>
    <div class="notice compact"><strong>Production run:</strong> ${escape(prediction.run_id || 'Not recorded')} · <strong>dataset:</strong> ${escape(shortFingerprint(prediction.dataset_fingerprint))} · <strong>challenger run:</strong> ${escape(challenger.experiment_run_id || 'Not recorded')}</div>
    <div class="detail-financial">
      <div><span>Production predicted cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div>
      <div><span>Challenger predicted cost overrun</span><strong>${fixed(challenger.predicted_cost_overrun)}%</strong></div>
      <div><span>Cost prediction difference</span><strong>${fixed(comparison.prediction_difference_pp)} pp</strong></div>
      <div><span>Challenger</span><strong>${escape(challenger.experiment_name || challenger.experiment_id || 'Experiment')}</strong></div>
      ${optionalAnchor}${optionalResidual}${trajectoryCoverage}
      <div><span>Production predicted delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div>
      ${delayPolicy}
      <div><span>Production predicted risk</span><strong>${escape(prediction.predicted_risk || 'N/A')}</strong></div>
    </div>
    <div class="notice compact"><strong>Leakage guard:</strong> the actual final outcome has not been sent to the browser yet.</div>
  </section>
  ${reveal ? `<section class="panel">
    <span class="kicker">Official outcome revealed once</span><h2>Which model was closer?</h2>
    <div class="detail-financial">
      <div><span>Actual final cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div>
      <div><span>Production cost error</span><strong>${fixed(reveal.production_cost_error_absolute_pp)} pp</strong></div>
      <div><span>Challenger cost error</span><strong>${fixed(reveal.experiment_cost_error_absolute_pp)} pp</strong></div>
      <div><span>Challenger cost improvement</span><strong>${escape(improvementLabel(reveal.individual_error_improvement_percentage))}</strong></div>
      ${revealDelay}
    </div>
    <div class="notice compact"><strong>Cost verdict:</strong> ${reveal.experiment_better_cost_for_project ?? reveal.experiment_better_for_project ? 'The challenger was closer.' : 'Production was at least as close.'} ${costOnly ? '<strong>Delay:</strong> Production model retained.' : (missing(reveal.experiment_better_delay_for_project) ? '' : `<strong>Delay verdict:</strong> ${reveal.experiment_better_delay_for_project ? 'The challenger was closer.' : 'Production was at least as close.'}`)}</div>
  </section>` : ''}`;
}

function productionPredictionCard(prediction, actual = null) {
  return `<section class="panel">
    <span class="kicker">Fresh production prediction · outcome withheld until reveal</span>
    <h2>${escape(prediction.project?.project_name || prediction.project?.project_id || 'Held-out project')}</h2>
    <div class="notice compact"><strong>Production run:</strong> ${escape(prediction.run_id || 'Not recorded')} · <strong>dataset:</strong> ${escape(shortFingerprint(prediction.dataset_fingerprint))}</div>
    <div class="detail-financial">
      <div><span>Predicted cost overrun</span><strong>${fixed(prediction.predicted_cost_overrun)}%</strong></div>
      <div><span>Predicted delay</span><strong>${fixed(prediction.predicted_delay_days)} days</strong></div>
      <div><span>Predicted risk</span><strong>${escape(prediction.predicted_risk || 'N/A')}</strong></div>
      <div><span>Risk probability</span><strong>${fixed(prediction.risk_probability_percentage)}%</strong></div>
    </div>
    <div class="notice compact"><strong>Leakage guard:</strong> the actual final outcome has not been sent to the browser yet.</div>
  </section>
  ${actual ? `<section class="panel">
    <span class="kicker">Official outcome revealed once</span><h2>Production-model result</h2>
    <div class="detail-financial">
      <div><span>Actual final cost overrun</span><strong>${fixed(actual.actual_cost_overrun)}%</strong></div>
      <div><span>Cost prediction error</span><strong>${fixed(actual.cost_error_absolute_pp)} pp</strong></div>
      <div><span>Actual final delay</span><strong>${fixed(actual.actual_delay_days)} days</strong></div>
      <div><span>Delay prediction error</span><strong>${fixed(actual.delay_error_absolute_days)} days</strong></div>
      <div><span>Actual risk</span><strong>${escape(actual.actual_risk || 'N/A')}</strong></div>
    </div>
  </section>` : ''}`;
}

export async function ModelSimulationPage(root) {
  const catalog = await api.simulationVersions();
  if (!catalog.lifecycle_data_available) {
    root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>Live Model Verification</h1></div></header><section class="panel"><div class="error-state">Official PAIMANA monthly lifecycle data is unavailable in this checkout.</div></section>`;
    return;
  }

  const years = catalog.data_years || [];
  if (!years.length) throw new Error('No identity-verified PAIMANA lifecycle years are available.');
  const experiments = catalog.comparison_experiments || [];
  const activeExperimentId = catalog.active_experiment_id || null;
  const activeExperimentName = catalog.active_experiment_name || null;
  const savedRun = api.getActiveLifecycleRun();
  const yearNumbers = years.map((item) => item.year);
  const savedStart = Number(savedRun?.start_year);
  const savedEnd = Number(savedRun?.end_year);
  const defaultStart = yearNumbers.includes(savedStart) ? savedStart : yearNumbers[0];
  const preferredEnd = yearNumbers.filter((year) => year <= 2015).at(-1);
  const defaultEnd = yearNumbers.includes(savedEnd) ? savedEnd : (preferredEnd || yearNumbers[Math.max(0, Math.floor(yearNumbers.length / 2) - 1)]);
  const challengerNotice = activeExperimentId
    ? `<strong>Active challenger:</strong> ${escape(activeExperimentName || activeExperimentId)}. The highest-numbered installed experiment adapter is selected automatically.`
    : '<strong>No challenger installed.</strong> You can still retrain the production lifecycle model for any valid year range and evaluate it on held-out projects.';
  const comparisonMode = Boolean(activeExperimentId);
  const trainAction = comparisonMode ? 'Retrain & Compare' : 'Retrain Production Model';
  const predictionAction = comparisonMode ? 'Generate Both Predictions' : 'Generate Production Prediction';

  root.innerHTML = `<header class="page-head"><div><span class="kicker">Judge-controlled historical lifecycle backtest</span><h1>${comparisonMode ? 'Production vs Experiment Verification' : 'Production Lifecycle Verification'}</h1><p>${comparisonMode ? 'Retrain the current lifecycle production model and the registered experiment challenger on one frozen PAIMANA evidence contract, then compare both on the same future projects.' : 'Retrain the current production lifecycle model for a selected year range, then evaluate it on unseen future projects.'}</p></div></header>
    <div class="notice">${challengerNotice}</div>
    <section class="panel">
      <div class="panel-head"><div><span class="kicker">Step 1</span><h2>${comparisonMode ? 'Retrain and compare' : 'Retrain production'}</h2></div></div>
      <div class="filters">
        <label>Training start year<select id="custom-start">${yearOptions(years, defaultStart)}</select></label>
        <label>Training end year<select id="custom-end">${yearOptions(years, defaultEnd)}</select></label>
        <button class="primary-btn" id="custom-train">${comparisonMode ? `Retrain & Compare vs ${escape(activeExperimentName || activeExperimentId)}` : trainAction}</button>
      </div>
      <div id="training-receipt" class="notice compact">${experiments.length ? 'No comparison has been trained in this browser session yet.' : 'No production lifecycle run has been trained in this browser session yet.'}</div>
      <div id="overall-comparison"></div>
    </section>
    <section class="panel">
      <div class="panel-head"><div><span class="kicker">Step 2</span><h2>Judge chooses one unseen future project</h2></div></div>
      <div class="filters">
        <label>Held-out completion year<select id="custom-test-year" disabled><option>${trainAction} first</option></select></label>
        <label>Official held-out project<select id="custom-project" disabled><option>Select a test year first</option></select></label>
        <button class="secondary-btn" id="random-project" disabled>Pick Random Project</button>
        <button class="primary-btn" id="custom-predict" disabled>${predictionAction}</button>
        <button class="secondary-btn" id="custom-reveal" disabled>Reveal Actual Outcome</button>
      </div>
      <div id="held-out-note" class="notice compact">Actual outcomes remain server-side until both predictions have been generated.</div>
    </section>
    <div id="custom-output"></div>`;

  const start = root.querySelector('#custom-start');
  const end = root.querySelector('#custom-end');
  const trainButton = root.querySelector('#custom-train');
  const receipt = root.querySelector('#training-receipt');
  const overallNode = root.querySelector('#overall-comparison');
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
  let activeExperiment = null;

  const resetPrediction = () => {
    prediction = null;
    revealButton.disabled = true;
    output.innerHTML = '';
  };

  const resetHeldOutState = (message = `${trainAction} first`) => {
    session = null;
    projectRows = [];
    activeExperiment = null;
    resetPrediction();
    testYear.innerHTML = `<option>${escape(message)}</option>`;
    testYear.disabled = true;
    project.innerHTML = '<option>Select a test year first</option>';
    project.disabled = true;
    randomButton.disabled = true;
    predictButton.disabled = true;
    revealButton.disabled = true;
    heldOutNote.innerHTML = comparisonMode
      ? '<strong>No fresh comparison session is active.</strong> Step 2 will unlock only after the requested production and challenger runs both finish successfully.'
      : '<strong>No fresh production session is active.</strong> Step 2 will unlock only after the requested year-range retraining finishes successfully.';
  };

  const loadProjects = async () => {
    if (!session || !testYear.value) return;
    resetPrediction();
    const response = comparisonMode
      ? await api.comparisonProjects(session.comparison_session_id || session.session_id, Number(testYear.value))
      : await api.customSimulationProjects(session.session_id, Number(testYear.value));
    projectRows = response.items || [];
    project.innerHTML = projectRows.length ? projectRows.map((row) => `<option value="${row.record_index}">${escape(row.project_id)} · ${escape(row.project_name)}</option>`).join('') : '<option>No projects available</option>';
    project.disabled = !projectRows.length;
    predictButton.disabled = !projectRows.length;
    randomButton.disabled = !projectRows.length;
    heldOutNote.innerHTML = `<strong>${projectRows.length} ${comparisonMode ? 'comparable' : 'held-out'} projects.</strong> ${escape(response.note || '')}`;
  };

  trainButton.addEventListener('click', async () => {
      overallNode.innerHTML = '';
      const startYear = Number(start.value);
      const endYear = Number(end.value);
      if (startYear > endYear) {
        resetHeldOutState('Invalid training range');
        receipt.innerHTML = '<div class="error-state">Training start year cannot be after training end year.</div>';
        return;
      }

      resetHeldOutState(`${trainAction} ${startYear}–${endYear}…`);
      trainButton.disabled = true;
      receipt.innerHTML = `<div class="loading">${comparisonMode ? `Retraining production and ${escape(activeExperimentName || activeExperimentId)}` : 'Retraining production'} for ${startYear}–${endYear} on one frozen lifecycle dataset. Step 2 has been cleared until this exact run finishes…</div>`;
      try {
        const result = comparisonMode
          ? await api.retrainAndCompare(startYear, endYear, activeExperimentId)
          : { production: await api.retrainModel(startYear, endYear) };
        const nextSession = comparisonMode
          ? result.session
          : await api.trainCustomSimulation(startYear, endYear, result.production?.run_id);
        const nextExperiment = comparisonMode ? result.experiment : null;
        if (!nextSession) throw new Error(`${trainAction} returned no judge session.`);

        const eligible = nextSession.eligible_test_years || [];
        const invalidYear = eligible.find((item) => Number(item.year) <= endYear);
        if (invalidYear) {
          throw new Error(`Leakage guard rejected held-out year ${invalidYear.year}; a ${startYear}–${endYear} training run may only offer years after ${endYear}.`);
        }

        session = nextSession;
        activeExperiment = nextExperiment;
        receipt.innerHTML = comparisonMode
          ? `<strong>Fresh comparison ready.</strong> Production run ${escape(result.production?.run_id || 'N/A')} · challenger run ${escape(activeExperiment?.run_id || 'N/A')} · dataset ${escape(shortFingerprint(result.production?.dataset_fingerprint))}.`
          : `<strong>Fresh production run ready.</strong> Production run ${escape(result.production?.run_id || 'N/A')} · dataset ${escape(shortFingerprint(result.production?.dataset_fingerprint))}.`;
        overallNode.innerHTML = comparisonMode ? overallCard(result.overall_comparison, activeExperiment) : '';
        api.setActiveLifecycleRun({
          window: result.production?.window,
          model_version: result.production?.model_version,
          run_id: result.production?.run_id,
          dataset_fingerprint: result.production?.dataset_fingerprint,
          start_year: startYear,
          end_year: endYear,
          receipt: result.production,
          session,
          experiment: activeExperiment,
          overall_comparison: comparisonMode ? result.overall_comparison : null,
        });
        testYear.innerHTML = eligible.length
          ? eligible.map((item) => `<option value="${item.year}">${item.year} · ${item.projects} projects</option>`).join('')
          : '<option>No eligible future years</option>';
        testYear.disabled = !eligible.length;
        if (eligible.length) await loadProjects();
        else heldOutNote.innerHTML = `<div class="error-state">The fresh ${comparisonMode ? 'comparison' : 'production run'} completed, but no eligible held-out projects exist after ${endYear}.</div>`;
      } catch (error) {
        resetHeldOutState(`${trainAction} failed`);
        overallNode.innerHTML = '';
        receipt.innerHTML = `<div class="error-state">${escape(error.message)}</div><p class="muted">No earlier held-out year or project is being reused. Retry the requested range to create a fresh session.</p>`;
      } finally {
        trainButton.disabled = false;
      }
    });

  testYear.addEventListener('change', () => loadProjects().catch((error) => {
    projectRows = [];
    project.innerHTML = '<option>No projects available</option>';
    project.disabled = true;
    predictButton.disabled = true;
    randomButton.disabled = true;
    heldOutNote.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
  }));
  project.addEventListener('change', resetPrediction);
  predictButton.addEventListener('click', async () => {
    if (!session || project.disabled) return;
    resetPrediction();
    try {
      prediction = comparisonMode
        ? await api.predictComparison(session.comparison_session_id || session.session_id, Number(project.value))
        : await api.predictCustomSimulation(session.session_id, Number(project.value));
      revealButton.disabled = false;
      output.innerHTML = comparisonMode ? predictionCard(prediction) : productionPredictionCard(prediction);
    } catch (error) {
      output.innerHTML = `<div class="error-state">${escape(error.message)}</div>`;
    }
  });
  randomButton.addEventListener('click', async () => {
    if (!projectRows.length) return;
    project.value = String(projectRows[Math.floor(Math.random() * projectRows.length)].record_index);
    predictButton.click();
  });
  revealButton.addEventListener('click', async () => {
    if (!session || !prediction) return;
    revealButton.disabled = true;
    try {
      const actual = comparisonMode
        ? await api.revealComparison(session.comparison_session_id || session.session_id, prediction.record_index)
        : await api.revealCustomSimulation(session.session_id, prediction.record_index);
      output.innerHTML = comparisonMode ? predictionCard(prediction, actual) : productionPredictionCard(prediction, actual);
    } catch (error) {
      output.innerHTML += `<div class="error-state">${escape(error.message)}</div>`;
      revealButton.disabled = false;
    }
  });
}
