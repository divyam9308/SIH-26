import { useEffect, useMemo, useState } from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AlertTriangle, ArrowLeft, Clock, IndianRupee, Radar, TrendingUp } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';
import { ApiError } from '../services/api';
import { getLifecycleForecast, getProject, getProjectForecast, getProjectPeers, getProjectWarnings } from '../services/projectService';
import type { ForecastResponse, LifecycleForecastResponse, PeerResponse, ProjectRecord, ShapFactor, CapabilityStatus, WarningResponse } from '../types/api';
import { displayRisk, inr, ProjectPanel, RiskChip, riskClass } from './Projects';
import '../styles/projects.css';
import { SAVED_WINDOW_STORAGE_KEY } from '../components/dashboard/FilterBar';

const unavailable = 'Not reported';
const formatDate = (value: string | null | undefined) => value ? new Date(`${value}T00:00:00`).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : unavailable;
const featureLabels: Record<string, string> = {
  approved_cost_cr: 'Approved project cost', revised_cost_cr: 'Revised project cost',
  physical_progress: 'Physical progress', physical_progress_pct: 'Physical progress',
  financial_progress: 'Financial progress', expenditure_ratio: 'Expenditure relative to approved cost',
  duration_ratio: 'Elapsed time relative to planned duration', schedule_slippage_days: 'Existing schedule slippage',
  cost_escalation_percentage: 'Recorded cost escalation', sector: 'Project sector', ministry: 'Line ministry',
  implementing_agency: 'Implementing agency',
};
const featureLabel = (value: string) => featureLabels[value] ?? value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const isUsableFactors = (factors: ShapFactor[]) => factors.some((factor) => factor.direction !== 'not available' || factor.impact !== 0);
const factorSentence = (factor: ShapFactor, target: 'cost' | 'delay' | 'risk', risk?: string | null) => {
  const label = featureLabel(factor.feature).toLowerCase();
  if (target === 'risk') return `${featureLabel(factor.feature)} ${factor.impact >= 0 ? 'pushed the model toward' : 'pushed the model away from'} the predicted ${risk ?? ''} risk category.`;
  const prediction = target === 'cost' ? 'cost-overrun estimate' : 'delay prediction';
  return `${featureLabel(factor.feature)} ${factor.impact >= 0 ? 'increased' : 'reduced'} the model's predicted ${prediction}.`;
};

function Field({ label, value, accent = '' }: { label: string; value: string; accent?: string }) {
  return <div className="detail-field"><dt>{label}</dt><dd className={accent} title={value}>{value}</dd></div>;
}

function FactorList({ title, factors, status, tone, target = 'cost', risk }: { title: string; factors: ShapFactor[] | undefined; status?: CapabilityStatus; tone: string; target?: 'cost' | 'delay' | 'risk'; risk?: string | null }) {
  if (!factors || !isUsableFactors(factors)) return <section className="risk-why-section"><h3>{title}</h3><p className="section-unavailable">{status?.reason ?? 'SHAP explanation unavailable for this model response.'}</p></section>;
  const sorted = [...factors].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const maximum = Math.max(...sorted.map((factor) => Math.abs(factor.impact)), 0.0001);
  return <section className="risk-why-section"><h3>{title}</h3><ol className="factor-list">{sorted.map((factor, index) => <li key={`${factor.feature}-${index}`}><div className="factor-top"><span><b className="factor-index">{String(index + 1).padStart(2, '0')}</b> {featureLabel(factor.feature)}</span><span className="factor-weight">{factor.impact > 0 ? '+' : ''}{factor.impact.toFixed(4)}</span></div><div className="factor-track"><span className={`tone-${tone}`} style={{ width: `${Math.abs(factor.impact) / maximum * 100}%` }} /></div><p>{factorSentence(factor, target, risk)}</p></li>)}</ol></section>;
}

export function ProjectDetail() {
  const { projectId = '' } = useParams();
  const navigate = useNavigate();
  const selectedWindow = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY) ?? undefined;
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [peers, setPeers] = useState<PeerResponse | null>(null);
  const [lifecycle, setLifecycle] = useState<LifecycleForecastResponse | null>(null);
  const [warnings, setWarnings] = useState<WarningResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [forecastStatus, setForecastStatus] = useState<string | null>(null);
  const [peerStatus, setPeerStatus] = useState<string | null>(null);
  const [warningStatus, setWarningStatus] = useState<string | null>(null);
  const [lifecycleStatus, setLifecycleStatus] = useState<string | null>(null);
  const [shapTab, setShapTab] = useState<'cost' | 'delay' | 'risk'>('cost');

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setProject(null); setForecast(null); setPeers(null); setLifecycle(null); setWarnings(null);
    setProjectError(null); setForecastStatus(null); setPeerStatus(null); setWarningStatus(null); setLifecycleStatus(null);
    Promise.allSettled([
      getProject(projectId, controller.signal, selectedWindow), getProjectForecast(projectId, controller.signal, selectedWindow),
      getProjectPeers(projectId, controller.signal, selectedWindow), getProjectWarnings(projectId, controller.signal, selectedWindow),
    ]).then(([projectResult, forecastResult, peersResult, warningsResult]) => {
      if (controller.signal.aborted) return;
      if (projectResult.status === 'fulfilled') setProject(projectResult.value); else setProjectError(projectResult.reason instanceof Error ? projectResult.reason.message : 'Project unavailable.');
      if (forecastResult.status === 'fulfilled') setForecast(forecastResult.value); else setForecastStatus(forecastResult.reason instanceof ApiError && forecastResult.reason.status === 409 ? `Prediction unavailable: ${forecastResult.reason.message}` : forecastResult.reason instanceof Error ? forecastResult.reason.message : 'Prediction unavailable.');
      if (peersResult.status === 'fulfilled') setPeers(peersResult.value); else setPeerStatus(peersResult.reason instanceof Error ? peersResult.reason.message : 'Peer benchmark unavailable.');
      if (warningsResult.status === 'fulfilled') setWarnings(warningsResult.value); else setWarningStatus(warningsResult.reason instanceof Error ? warningsResult.reason.message : 'Warning events unavailable.');
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
  const progress = project.physical_progress_pct;
  const financialProgress = project.financial_progress_pct;

  const shapConfig = {
    cost: { label: 'Cost', title: 'Cost SHAP factors', factors: forecast?.cost_factors, status: forecast?.cost_explanation_status },
    delay: { label: 'Delay', title: 'Delay SHAP factors', factors: forecast?.delay_factors, status: forecast?.delay_explanation_status },
    risk: { label: 'Risk', title: 'Risk SHAP factors', factors: forecast?.risk_factors, status: forecast?.risk_explanation_status },
  }[shapTab];

  return <div className="projects-page project-detail">
    <button className="back-to-register" onClick={() => navigate('/projects')}><ArrowLeft size={14} /> Back to risk register</button>
    <section className="detail-hero compact-project-header"><div><p className="projects-eyebrow">Project Intelligence</p><h1>{project.project_name}</h1><p className="detail-meta">{project.project_code} · {project.sector} · {project.implementing_agency ?? unavailable}</p><p className="detail-provenance">Dataset {project.snapshot_date} · Model {forecast?.model_version ?? 'Unavailable'} · Inference {forecast ? new Date(forecast.inference_timestamp).toLocaleString('en-IN') : 'Unavailable'}</p></div></section>
    {forecastStatus && <div className="partial-data-banner"><AlertTriangle size={16} />{forecastStatus}. Project information remains available.</div>}

    <section className="prediction-summary" aria-label="Executive prediction summary">
      <article className="prediction-card"><p>Predicted Cost Overrun</p><strong className={tone}>{forecast ? `${forecast.predicted_cost_overrun_percentage > 0 ? '+' : ''}${forecast.predicted_cost_overrun_percentage.toFixed(1)}%` : 'Unavailable'}</strong><span>{forecast ? `${inr(forecast.predicted_cost_overrun_amount_cr)} estimated overrun` : unavailable}</span>{forecast && <small>Predicted final cost {inr(forecast.predicted_final_cost_cr)}</small>}</article>
      <article className="prediction-card"><p>Predicted Time Overrun</p><strong className={tone}>{forecast ? `${forecast.predicted_delay_months.toFixed(1)} months` : 'Unavailable'}</strong><span>{forecast ? `${forecast.predicted_delay_days.toFixed(0)} days` : unavailable}</span>{forecast?.predicted_completion_date && <small>Predicted completion {formatDate(forecast.predicted_completion_date)}</small>}</article>
      <article className="prediction-card"><p>Overall Risk</p>{category ? <RiskChip level={category} /> : <strong>Unavailable</strong>}<span>{forecast ? `${hasRiskProbability ? 'Implementation risk score' : 'Calibrated risk severity'} ${forecast.risk_score.toFixed(1)}/100` : unavailable}</span><small>{hasRiskProbability ? 'Risk probability is reported separately.' : 'Severity is not classifier confidence.'}</small></article>
    </section>

    <ProjectPanel className="evidence-panel" title="Model Evidence" subtitle="Model Evidence · Local SHAP — project-level inputs that pushed each prediction higher or lower.">
      <div className="evidence-tabs" role="tablist">{(['cost', 'delay', 'risk'] as const).map((tab) => <button key={tab} role="tab" aria-selected={shapTab === tab} className={shapTab === tab ? 'active' : ''} onClick={() => setShapTab(tab)}>{shapConfig && tab === shapTab ? shapConfig.label : tab[0].toUpperCase() + tab.slice(1)}</button>)}</div>
      <div className="evidence-body"><FactorList title={shapConfig.title} factors={shapConfig.factors} status={shapConfig.status} tone={tone} target={shapTab} risk={category} /></div>
    </ProjectPanel>

    <ProjectPanel className="operational-panel" title="Operational Drivers" subtitle="Observed warning signals derived directly from available PAIMANA project records.">
      <div className="operational-panel-body">{forecast?.operational_drivers.length ? <ul className="operational-driver-list">{forecast.operational_drivers.map((driver) => <li key={driver.type}><div className="factor-top"><b>{driver.label}</b><span className="factor-weight">Observed signal</span></div><p>{driver.evidence}</p></li>)}</ul> : <p className="section-unavailable">No verified operational drivers were available for this project snapshot.</p>}</div>
    </ProjectPanel>

    <ProjectPanel className="future-integration" title="Future Integration — Administrative Cause Intelligence" subtitle="Potential future integration with additional authorised project-monitoring data." action={<span className="proposed-badge">Proposed</span>}>
      <div className="future-body"><div className="future-categories">{[['Land acquisition', 'Authorised land / ministry records'], ['Environmental clearances', 'Authorised clearance systems'], ['Litigation', 'Ministry or project legal-status records'], ['Contractor execution', 'Structured execution reports'], ['Funding / sanction', 'Administrative sanction records']].map(([name, source]) => <div key={name}><b>{name}</b><span>Potential source: {source}</span></div>)}</div><p>Conceptual future capability — these administrative cause fields are not present in the current PAIMANA-derived dataset and do not affect the predictions shown above.</p></div>
    </ProjectPanel>

    <div className="detail-grid"><ProjectPanel title="Project Information" className="information"><dl className="detail-fields"><Field label="Project name" value={project.project_name} /><Field label="Project code" value={project.project_code} /><Field label="Sector" value={project.sector} /><Field label="Line ministry / department" value={project.ministry ?? unavailable} /><Field label="Implementing agency" value={project.implementing_agency ?? unavailable} /></dl></ProjectPanel>
      <ProjectPanel title="Cost Intelligence" subtitle="₹ crore" className="span-two" action={<IndianRupee size={16} color="var(--p-muted-foreground)" />}><dl className="detail-fields"><Field label="Original approved cost" value={inr(project.original_cost_cr)} /><Field label="Latest revised cost" value={inr(project.revised_cost_cr)} /><Field label="Cumulative expenditure" value={inr(project.expenditure_cr)} /><Field label="Predicted final cost" value={forecast ? inr(forecast.predicted_final_cost_cr) : 'Unavailable'} accent={forecast ? tone : ''} /><Field label="Predicted cost overrun" value={forecast ? `${inr(forecast.predicted_cost_overrun_amount_cr)} (${forecast.predicted_cost_overrun_percentage > 0 ? '+' : ''}${forecast.predicted_cost_overrun_percentage.toFixed(1)}%)` : 'Unavailable'} accent={forecast ? tone : ''} /></dl>
        <div className="detail-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={costSeries} margin={{ left: 8, right: 8, top: 8 }}><defs><linearGradient id={chartGradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--p-primary)" stopOpacity={0.35} /><stop offset="100%" stopColor="var(--p-primary)" stopOpacity={0.02} /></linearGradient></defs><CartesianGrid stroke="var(--p-border)" vertical={false} /><XAxis dataKey="stage" tick={{ fontSize: 11 }} stroke="var(--p-muted-foreground)" /><YAxis tick={{ fontSize: 11 }} stroke="var(--p-muted-foreground)" width={64} /><Tooltip formatter={(value) => [inr(Number(value)), 'Cost']} /><Area type="monotone" dataKey="value" stroke="var(--p-primary)" strokeWidth={2} fill={`url(#${chartGradient})`} /></AreaChart></ResponsiveContainer></div>
        {forecast && <p className="range-note">{forecast.expected_range ? <>Expected cost overrun range: {forecast.expected_range.cost_overrun_percentage.p10}% to {forecast.expected_range.cost_overrun_percentage.p90}% · confidence {forecast.model_confidence_percentage === null ? 'Unavailable' : `${forecast.model_confidence_percentage}%`} · {forecast.confidence_calibration_status.replaceAll('_', ' ')}</> : 'Uncertainty interval unavailable for this prediction.'}</p>}
      </ProjectPanel></div>

    <div className="detail-grid"><ProjectPanel title="Timeline Intelligence" className="span-two" action={<Clock size={16} color="var(--p-muted-foreground)" />}><dl className="detail-fields"><Field label="Project start" value={unavailable} /><Field label="Original completion" value={formatDate(project.original_end_date)} /><Field label="Latest revised completion" value={formatDate(project.revised_end_date)} /><Field label="Predicted completion" value={forecast ? formatDate(forecast.predicted_completion_date) : 'Unavailable'} accent={forecast ? tone : ''} /><Field label="Predicted time overrun" value={forecast ? `${forecast.predicted_delay_months.toFixed(1)} months` : 'Unavailable'} accent={forecast ? tone : ''} /></dl>{forecast?.expected_range && <p className="range-note">Expected delay range: {forecast.expected_range.delay_days.p10.toFixed(1)} to {forecast.expected_range.delay_days.p90.toFixed(1)} days.</p>}</ProjectPanel>
      <ProjectPanel title="Progress & Risk" action={<TrendingUp size={16} color="var(--p-muted-foreground)" />}><div className="progress-content">{[["Physical progress", progress, 'primary'], ["Financial progress", financialProgress, 'financial'], [hasRiskProbability ? "Implementation risk score" : "Calibrated risk severity", forecast?.risk_score ?? null, tone]].map(([label, value, color]) => <div key={String(label)}><div className="progress-label"><span>{label}</span><b className={color === tone ? tone : ''}>{value === null ? unavailable : `${Number(value).toFixed(1)}%`}</b></div><div className="progress-track">{value !== null && <span className={`tone-${color}`} style={{ width: `${Math.max(0, Math.min(100, Number(value)))}%` }} />}</div></div>)}<p className="progress-note">Missing progress values are preserved as not reported; they are not converted to zero.</p></div></ProjectPanel></div>

    <ProjectPanel className="why-risk" title="Additional project context"><div className="risk-why-grid"><section className="risk-why-section"><div className="factor-top"><h3>Peer benchmark</h3><Radar size={16} color="var(--p-muted-foreground)" /></div>{peers ? <dl className="peer-metrics"><Field label="Comparable projects" value={String(peers.peer_count)} /><Field label="Median approved cost" value={inr(peers.medians.original_cost_cr)} /><Field label="Median cost escalation" value={peers.medians.cost_escalation_pct === null ? unavailable : `${peers.medians.cost_escalation_pct}%`} /><Field label="Median schedule extension" value={peers.medians.schedule_extension_days === null ? unavailable : `${peers.medians.schedule_extension_days} days`} /></dl> : <p className="section-unavailable">{peerStatus ?? 'Peer benchmark unavailable.'}</p>}</section>
      <section className="risk-why-section"><h3>Early warning signals</h3>{warnings?.available ? (warnings.items.length ? <ol className="factor-list">{warnings.items.map((warning) => <li key={`${warning.type}-${warning.date}`}><div className="factor-top"><span>{featureLabel(warning.type)}</span><span className="factor-weight">{warning.severity}</span></div><p>{warning.message}</p></li>)}</ol> : <p className="section-unavailable">No snapshot-change warning events occurred at this evaluation snapshot.</p>) : <p className="section-unavailable">{warnings?.reason ?? warningStatus ?? 'Warning events unavailable.'}</p>}</section>
      <section className="risk-why-section"><h3>Lifecycle trajectory</h3>{lifecycle ? <dl className="peer-metrics"><Field label="Lifecycle model" value={lifecycle.model_version} /><Field label="Official snapshots" value={String(lifecycle.history_snapshots)} /><Field label="Lifecycle risk" value={lifecycle.risk_level} /><Field label="Provenance" value={lifecycle.provenance.verified ? 'Verified' : 'Unverified'} /></dl> : <p className="section-unavailable">{lifecycleStatus ?? 'Lifecycle history unavailable.'}</p>}</section>
    </div></ProjectPanel>
  </div>;
}
