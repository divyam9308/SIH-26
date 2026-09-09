import { useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AlertTriangle, ArrowLeft, IndianRupee, Radar } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';
import { ApiError } from '../services/api';
import { getLifecycleForecast, getProject, getProjectForecast, getProjectPeers } from '../services/projectService';
import { getEarlyWarnings } from '../services/earlyWarningsService';
import type { ForecastResponse, LifecycleForecastResponse, PeerResponse, ProjectRecord, ShapFactor, CapabilityStatus, ExplanationSummary } from '../types/api';
import type { EarlyWarningRow } from '../types/earlyWarnings';
import { displayRisk, inr, ProjectPanel, RiskChip, riskClass } from './Projects';
import { shapExplanationSubject, shapFeatureLabel } from '../lib/shapFeatureLabels';
import '../styles/projects.css';
import { SAVED_WINDOW_STORAGE_KEY } from '../components/dashboard/FilterBar';
import { predictionActualError } from '../lib/predictionComparison';

const unavailable = 'Not reported';
const percentage = (value: number | null | undefined) => value == null ? 'Unavailable' : `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
const days = (value: number | null | undefined) => value == null ? 'Unavailable' : `${value.toFixed(0)} days`;
const warningDate = (value: string | null) => value ? new Intl.DateTimeFormat('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }).format(new Date(`${value}T00:00:00`)) : 'Not reported';
const comparison = (predicted: number | null | undefined, actual: number | null | undefined) => {
  const error = predictionActualError(predicted, actual);
  return error === null ? 'Unavailable' : `${error.toFixed(1)}%`;
};
const featureLabel = shapFeatureLabel;
const isUsableFactors = (factors: ShapFactor[]) => factors.some((factor) => factor.direction !== 'not available' || factor.impact !== 0);
const factorSentence = (factor: ShapFactor, target: 'cost' | 'delay' | 'risk', risk?: string | null) => {
  const subject = shapExplanationSubject(factor.feature);
  if (target === 'risk') return `${subject} ${factor.impact >= 0 ? 'pushed the prediction toward' : 'pushed the prediction away from'} the ${risk ?? 'predicted'} risk category.`;
  const prediction = target === 'cost' ? 'predicted cost overrun' : 'predicted delay';
  return `${subject} ${factor.impact >= 0 ? 'increased' : 'reduced'} the ${prediction}.`;
};

function Field({ label, value, accent = '' }: { label: string; value: string; accent?: string }) {
  return <div className="detail-field"><dt>{label}</dt><dd className={accent} title={value}>{value}</dd></div>;
}

function FactorList({ title, factors, status, tone, target = 'cost', risk }: { title: string; factors: ShapFactor[] | undefined; status?: CapabilityStatus; tone: string; target?: 'cost' | 'delay' | 'risk'; risk?: string | null }) {
  if (!factors || !isUsableFactors(factors)) return <section className="risk-why-section"><h3>{title}</h3><p className="section-unavailable">{status?.reason ?? 'SHAP explanation unavailable for this model response.'}</p></section>;
  const sorted = [...factors].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const maximum = Math.max(...sorted.map((factor) => Math.abs(factor.impact)), 0.0001);
  return <section className="risk-why-section"><h3>{title}</h3><div className="factor-axis" aria-label="Contribution direction: reductions left of zero, increases right of zero"><span>Reduces prediction</span><b>0</b><span>Increases prediction</span></div><ol className="factor-list">{sorted.map((factor, index) => { const width = Math.abs(factor.impact) / maximum * 50; const left = factor.impact < 0 ? 50 - width : 50; return <li key={`${factor.feature}-${index}`}><div className="factor-top"><span><b className="factor-index">{String(index + 1).padStart(2, '0')}</b> {featureLabel(factor.feature)}</span><span className="factor-weight">{factor.impact > 0 ? '+' : ''}{factor.impact.toFixed(4)}</span></div><div className="factor-track factor-track-centered"><i /><span className={`tone-${tone} ${factor.impact < 0 ? 'factor-bar-negative' : 'factor-bar-positive'}`} style={{ width: `${width}%`, left: `${left}%` }} /></div><p>{factorSentence(factor, target, risk)}</p></li>; })}</ol></section>;
}

const outputLabel = (summary: ExplanationSummary, target: 'cost' | 'delay' | 'risk') => summary.output === 'predicted_class_probability' ? `Predicted ${summary.predicted_class ?? 'selected'}-class probability` : target === 'cost' ? 'Predicted cost overrun' : 'Predicted delay';

function PredictionDecomposition({ summary, target }: { summary?: ExplanationSummary | null; target: 'cost' | 'delay' | 'risk' }) {
  if (!summary?.available || summary.base_value === null || summary.prediction === null || summary.net_feature_impact === null) return null;
  const precision = target === 'risk' ? 4 : 2;
  const number = (value: number) => `${value > 0 ? '+' : ''}${value.toFixed(precision)}`;
  const riskNote = summary.output === 'predicted_class_probability' ? ' This is the frozen classifier output, separate from the headline calibrated delay-severity policy.' : '';
  return <section className="prediction-decomposition" aria-label="Prediction decomposition"><div><span>Historical reference</span><b>{summary.base_value.toFixed(precision)}</b></div><div><span>Net project-specific impact</span><b>{number(summary.net_feature_impact)}</b></div><div><span>{outputLabel(summary, target)}</span><b>{summary.prediction.toFixed(precision)}</b></div>{Math.abs(summary.other_features_impact ?? 0) > 0.000001 && <p>Top displayed factors contribute {number(summary.displayed_factors_impact ?? 0)}; other model features contribute {number(summary.other_features_impact ?? 0)}.</p>}<details><summary>How calculated?</summary><p>{summary.reference_description} The baseline is a local model reference, not an approved cost or final outcome. Factor impacts add to the net project-specific impact; {outputLabel(summary, target).toLowerCase()} = baseline + net impact.{riskNote}</p></details></section>;
}

export function ProjectDetail() {
  const { projectId = '' } = useParams();
  const navigate = useNavigate();
  const selectedWindow = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY) ?? undefined;
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [peers, setPeers] = useState<PeerResponse | null>(null);
  const [lifecycle, setLifecycle] = useState<LifecycleForecastResponse | null>(null);
  const [warnings, setWarnings] = useState<EarlyWarningRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [forecastStatus, setForecastStatus] = useState<string | null>(null);
  const [peerStatus, setPeerStatus] = useState<string | null>(null);
  const [warningStatus, setWarningStatus] = useState<string | null>(null);
  const [lifecycleStatus, setLifecycleStatus] = useState<string | null>(null);
  const [shapTab, setShapTab] = useState<'cost' | 'delay' | 'risk'>('cost');

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setProject(null); setForecast(null); setPeers(null); setLifecycle(null); setWarnings([]);
    setProjectError(null); setForecastStatus(null); setPeerStatus(null); setWarningStatus(null); setLifecycleStatus(null);
    Promise.allSettled([
      getProject(projectId, controller.signal, selectedWindow), getProjectForecast(projectId, controller.signal, selectedWindow),
      getProjectPeers(projectId, controller.signal, selectedWindow), getEarlyWarnings({ projectId, window: '2001_2022' }, controller.signal),
    ]).then(([projectResult, forecastResult, peersResult, warningsResult]) => {
      if (controller.signal.aborted) return;
      if (projectResult.status === 'fulfilled') setProject(projectResult.value); else setProjectError(projectResult.reason instanceof Error ? projectResult.reason.message : 'Project unavailable.');
      if (forecastResult.status === 'fulfilled') setForecast(forecastResult.value); else setForecastStatus(forecastResult.reason instanceof ApiError && forecastResult.reason.status === 409 ? `Prediction unavailable: ${forecastResult.reason.message}` : forecastResult.reason instanceof Error ? forecastResult.reason.message : 'Prediction unavailable.');
      if (peersResult.status === 'fulfilled') setPeers(peersResult.value); else setPeerStatus(peersResult.reason instanceof Error ? peersResult.reason.message : 'Peer benchmark unavailable.');
      if (warningsResult.status === 'fulfilled') setWarnings(warningsResult.value.warnings); else setWarningStatus(warningsResult.reason instanceof Error ? warningsResult.reason.message : 'Warning events unavailable.');
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    // The lifecycle artifact can need a moment to warm after a local backend restart.
    // Retry only transport/server failures: a 404/409 remains an honest unavailable state.
    const loadLifecycle = async (attempt = 0): Promise<void> => {
      try {
        const result = await getLifecycleForecast(projectId, controller.signal, selectedWindow);
        if (!controller.signal.aborted) setLifecycle(result);
      } catch (reason) {
        if (controller.signal.aborted) return;
        const retryable = reason instanceof ApiError && (reason.status === 0 || reason.status >= 500);
        if (retryable && attempt < 2) {
          window.setTimeout(() => { void loadLifecycle(attempt + 1); }, (attempt + 1) * 750);
          return;
        }
        setLifecycleStatus(reason instanceof ApiError && [404, 409].includes(reason.status) ? 'Lifecycle history unavailable for this project.' : reason instanceof Error ? reason.message : 'Lifecycle history unavailable.');
      }
    };
    void loadLifecycle();
    return () => controller.abort();
  }, [projectId, selectedWindow]);

  const costSeries = useMemo(() => project ? [
    { stage: 'Approved', value: project.original_cost_cr },
    ...(project.revised_cost_cr === null ? [] : [{ stage: 'Revised', value: project.revised_cost_cr }]),
    ...(project.expenditure_cr === null ? [] : [{ stage: 'Expenditure', value: project.expenditure_cr }]),
    ...(forecast ? [{ stage: 'Predicted final', value: forecast.predicted_final_cost_cr }] : []),
  ] : [], [project, forecast]);

  if (loading && !project) return <div className="projects-page project-detail"><ProjectPanel className="not-found"><p>Loading real project data and predictions…</p></ProjectPanel></div>;
  if (!project) return <div className="projects-page project-detail"><ProjectPanel className="not-found"><h1>Project unavailable</h1><p>{projectError ?? 'This project is not in the real PAIMANA project register.'}</p><button className="back-to-register" onClick={() => navigate('/projects')}><ArrowLeft size={14} /> Back to risk register</button></ProjectPanel></div>;

  const category = forecast ? displayRisk(forecast.risk_level) : null;
  const tone = category ? riskClass(category) : 'medium';
  const hasRiskProbability = forecast?.risk_probability_percentage !== null && forecast?.risk_probability_percentage !== undefined;
  const chartGradient = `cost-fill-${project.project_code}`;

  const shapConfig = {
    cost: { label: 'Cost', title: 'Cost SHAP factors', factors: forecast?.cost_factors, status: forecast?.cost_explanation_status, summary: forecast?.cost_explanation_summary },
    delay: { label: 'Delay', title: 'Delay SHAP factors', factors: forecast?.delay_factors, status: forecast?.delay_explanation_status, summary: forecast?.delay_explanation_summary },
    risk: { label: 'Risk', title: 'Risk SHAP factors', factors: forecast?.risk_factors, status: forecast?.risk_explanation_status, summary: forecast?.risk_explanation_summary },
  }[shapTab];

  return <div className="projects-page project-detail">
    <button className="back-to-register" onClick={() => navigate('/projects')}><ArrowLeft size={14} /> Back to risk register</button>
    <section className="detail-hero compact-project-header"><div><p className="projects-eyebrow">Project Intelligence</p><h1>{project.project_name}</h1><p className="detail-meta">{project.project_code} · {project.sector} · {project.implementing_agency ?? unavailable}</p></div></section>
    {forecastStatus && <div className="partial-data-banner"><AlertTriangle size={16} />{forecastStatus}. Project information remains available.</div>}

    <section className="prediction-summary" aria-label="Executive prediction summary">
      <article className="prediction-card prediction-comparison-card"><p>Cost Overrun</p><div className="prediction-pair"><div><span>Predicted Cost Overrun</span><strong className={tone}>{percentage(forecast?.predicted_cost_overrun_percentage)}</strong></div><div><span>Actual Cost Overrun</span><strong>{percentage(project.cost_escalation_pct)}</strong></div></div><small>Error: {comparison(forecast?.predicted_cost_overrun_percentage, project.cost_escalation_pct)}</small></article>
      <article className="prediction-card prediction-comparison-card"><p>Time Overrun</p><div className="prediction-pair"><div><span>Predicted Time Overrun</span><strong className={tone}>{days(forecast?.predicted_delay_days)}</strong></div><div><span>Actual Time Overrun</span><strong>{days(project.schedule_extension_days)}</strong></div></div><small>Error: {comparison(forecast?.predicted_delay_days, project.schedule_extension_days)}</small></article>
      <article className="prediction-card"><p>Overall Risk</p>{category ? <RiskChip level={category} /> : <strong>Unavailable</strong>}<span>{forecast ? `${hasRiskProbability ? 'Implementation risk score' : 'Calibrated risk severity'} ${forecast.risk_score.toFixed(1)}/100` : unavailable}</span>{hasRiskProbability && <small>Risk probability {forecast.risk_probability_percentage.toFixed(1)}%</small>}</article>
    </section>

    <ProjectPanel className="evidence-panel" title="Model Evidence" subtitle="Model Evidence · Local SHAP — project-level inputs that pushed each prediction higher or lower.">
      <div className="evidence-tabs" role="tablist">{(['cost', 'delay', 'risk'] as const).map((tab) => <button key={tab} role="tab" aria-selected={shapTab === tab} className={shapTab === tab ? 'active' : ''} onClick={() => setShapTab(tab)}>{shapConfig && tab === shapTab ? shapConfig.label : tab[0].toUpperCase() + tab.slice(1)}</button>)}</div>
      <div className="evidence-body"><PredictionDecomposition summary={shapConfig.summary} target={shapTab} /><FactorList title={shapConfig.title} factors={shapConfig.factors} status={shapConfig.status} tone={tone} target={shapTab} risk={shapTab === 'risk' ? shapConfig.summary?.predicted_class ?? category : category} /></div>
    </ProjectPanel>

    <ProjectPanel className="operational-panel" title="Operational Drivers" subtitle="Observed warning signals derived directly from available PAIMANA project records.">
      <div className="operational-panel-body">{forecast?.operational_drivers.length ? <ul className="operational-driver-list">{forecast.operational_drivers.map((driver) => <li key={driver.type}><div className="factor-top"><b>{driver.label}</b><span className="factor-weight">Observed signal</span></div><p>{driver.evidence}</p></li>)}</ul> : <p className="section-unavailable">No material operational warning signal was identified from the available PAIMANA records for this snapshot.</p>}</div>
    </ProjectPanel>

    <ProjectPanel className="future-integration" title="Future Integration — Administrative Cause Intelligence" subtitle="Potential future integration with additional authorised project-monitoring data." action={<span className="proposed-badge">Proposed</span>}>
      <div className="future-body"><div className="future-categories">{[['Land acquisition', 'Authorised land / ministry records'], ['Environmental clearances', 'Authorised clearance systems'], ['Litigation', 'Ministry or project legal-status records'], ['Contractor execution', 'Structured execution reports'], ['Funding / sanction', 'Administrative sanction records']].map(([name, source]) => <div key={name}><b>{name}</b><span>Potential source: {source}</span></div>)}</div><p>Conceptual future capability — these administrative cause fields are not present in the current PAIMANA-derived dataset and do not affect the predictions shown above.</p></div>
    </ProjectPanel>

    <div className="detail-grid"><ProjectPanel title="Project Information" className="information"><dl className="detail-fields"><Field label="Project name" value={project.project_name} /><Field label="Project code" value={project.project_code} /><Field label="Sector" value={project.sector} /><Field label="Line ministry / department" value={project.ministry ?? unavailable} /><Field label="Implementing agency" value={project.implementing_agency ?? unavailable} /></dl></ProjectPanel>
      <ProjectPanel title="Cost Intelligence" subtitle="₹ crore" className="span-two" action={<IndianRupee size={16} color="var(--p-muted-foreground)" />}><dl className="detail-fields"><Field label="Original approved cost" value={inr(project.original_cost_cr)} /><Field label="Latest revised cost" value={inr(project.revised_cost_cr)} /><Field label="Cumulative expenditure" value={inr(project.expenditure_cr)} /><Field label="Predicted final cost" value={forecast ? inr(forecast.predicted_final_cost_cr) : 'Unavailable'} accent={forecast ? tone : ''} /><Field label="Predicted cost overrun" value={forecast ? `${inr(forecast.predicted_cost_overrun_amount_cr)} (${forecast.predicted_cost_overrun_percentage > 0 ? '+' : ''}${forecast.predicted_cost_overrun_percentage.toFixed(1)}%)` : 'Unavailable'} accent={forecast ? tone : ''} /></dl>
        <div className="detail-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={costSeries} margin={{ left: 8, right: 8, top: 8 }}><defs><linearGradient id={chartGradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--p-primary)" stopOpacity={0.35} /><stop offset="100%" stopColor="var(--p-primary)" stopOpacity={0.02} /></linearGradient></defs><CartesianGrid stroke="var(--p-border)" vertical={false} /><XAxis dataKey="stage" tick={{ fontSize: 11 }} stroke="var(--p-muted-foreground)" /><YAxis tick={{ fontSize: 11 }} stroke="var(--p-muted-foreground)" width={64} /><Tooltip formatter={(value) => [inr(Number(value)), 'Cost']} /><Area type="monotone" dataKey="value" stroke="var(--p-primary)" strokeWidth={2} fill={`url(#${chartGradient})`} /></AreaChart></ResponsiveContainer></div>
        {forecast?.expected_range && <p className="range-note">Expected cost overrun range: {forecast.expected_range.cost_overrun_percentage.p10}% to {forecast.expected_range.cost_overrun_percentage.p90}% · confidence {forecast.model_confidence_percentage === null ? 'Unavailable' : `${forecast.model_confidence_percentage}%`} · {forecast.confidence_calibration_status.replaceAll('_', ' ')}</p>}
      </ProjectPanel></div>

    <ProjectPanel className="why-risk" title="Project context"><div className="context-upper-grid"><section className="risk-why-section"><div className="factor-top"><h3>Peer Benchmark</h3><Radar size={16} color="var(--p-muted-foreground)" /></div>{peers ? <dl className="peer-metrics"><Field label="Comparable projects" value={String(peers.peer_count)} /><Field label="Median approved cost" value={inr(peers.medians.original_cost_cr)} /><Field label="Median cost escalation" value={peers.medians.cost_escalation_pct === null ? unavailable : `${peers.medians.cost_escalation_pct}%`} /><Field label="Median schedule extension" value={peers.medians.schedule_extension_days === null ? unavailable : `${peers.medians.schedule_extension_days} days`} /></dl> : <p className="section-unavailable">{peerStatus ?? 'Peer benchmark unavailable.'}</p>}</section>
      <section className="risk-why-section"><h3>Lifecycle Prediction</h3>{lifecycle ? <dl className="peer-metrics"><Field label="Lifecycle model" value={lifecycle.model_version} /><Field label="Official snapshots" value={String(lifecycle.history_snapshots)} /><Field label="Lifecycle risk" value={lifecycle.risk_level} /><Field label="Provenance" value={lifecycle.provenance.verified ? 'Verified' : 'Unverified'} /></dl> : <p className="section-unavailable">{lifecycleStatus ?? 'Lifecycle history unavailable.'}</p>}</section></div></ProjectPanel>

    <ProjectPanel className="early-warning-panel" title="Early Warning Signals" subtitle="The same real warning records shown in the Active Warning Queue, matched by project code."><div className="early-warning-body">{warnings.length ? <ol className="project-warning-list">{warnings.map((warning) => <li key={`${warning.project_id}-${warning.snapshot_date}`}><div className="factor-top"><b>{warning.warning_status}</b><span className={`project-warning-severity ${warning.current_risk.level.toLowerCase()}`}>{warning.current_risk.level} · {warning.current_risk.score.toFixed(0)}/100</span></div><p>{warning.trigger}</p><dl className="project-warning-meta"><div><dt>Detected</dt><dd>{warningDate(warning.snapshot_date)}</dd></div><div><dt>First appeared</dt><dd>{warningDate(warning.first_appeared)}</dd></div><div><dt>Risk change</dt><dd>{warning.risk_change === null ? 'Not available' : `${warning.risk_change > 0 ? '+' : ''}${warning.risk_change.toFixed(1)} points`}</dd></div></dl></li>)}</ol> : <p className="section-unavailable">{warningStatus ?? 'No active early warning signals for this project.'}</p>}</div></ProjectPanel>
  </div>;
}
