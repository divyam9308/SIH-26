import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { AlertTriangle, BarChart3, CalendarDays, CheckCircle2, Clock3, Info, RefreshCw, TrendingUp } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { getPredictionAccuracyData, type PredictionAccuracyData, type ValidationRow } from '../services/predictionAccuracyService';
import { SAVED_WINDOW_STORAGE_KEY, SAVED_WINDOWS } from '../components/dashboard/FilterBar';
import '../styles/predictionAccuracy.css';

const blue = '#2563eb';
const orange = '#f97316';
const riskNames = ['Low', 'Moderate', 'High', 'Critical'] as const;
const riskLabel = (value: number | string) => typeof value === 'number' ? riskNames[Math.max(0, Math.min(3, value))] : `${value[0]}${value.slice(1).toLowerCase()}`;
const metricValue = (value: number | undefined) => value === undefined ? 'Unavailable' : value.toFixed(3);

function Panel({ title, children, className = '', action }: { title: string; children: ReactNode; className?: string; action?: ReactNode }) {
  return <section className={`pa-panel ${className}`}><header><h2>{title}</h2>{action}</header>{children}</section>;
}
function SummaryCard({ icon, title, footer, children, accent = 'blue' }: { icon: ReactNode; title: string; footer: string; children: ReactNode; accent?: 'blue' | 'orange' }) {
  return <article className="pa-summary-card"><div className={`pa-summary-icon ${accent}`}>{icon}</div><div><h3>{title}</h3><div className="pa-summary-values">{children}</div><p>{footer}</p></div></article>;
}
function Value({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><b>{value}</b></div>; }

function ScatterPanel({ title, rows, kind }: { title: string; rows: ValidationRow[]; kind: 'cost' | 'delay' }) {
  const cost = kind === 'cost';
  const points = rows.map((row) => ({ name: row.project_name, actual: cost ? row.actual_cost_overrun : row.actual_delay_days, predicted: cost ? row.predicted_cost_overrun : row.predicted_delay_days }));
  const values = points.flatMap((point) => [point.actual, point.predicted]);
  const low = Math.min(0, ...values); const high = Math.max(1, ...values);
  return <Panel title={title} className="pa-chart-panel"><div className="pa-chart scatter-chart"><ResponsiveContainer><ScatterChart margin={{ top: 12, right: 12, bottom: 18, left: 4 }}><CartesianGrid stroke="#e3eaf3" /><XAxis dataKey="actual" type="number" domain={[low, high]} tick={{ fontSize: 9 }} /><YAxis dataKey="predicted" type="number" domain={[low, high]} tick={{ fontSize: 9 }} /><ReferenceLine segment={[{ x: low, y: low }, { x: high, y: high }]} stroke="#94a3b8" strokeDasharray="4 3" /><Tooltip /><Scatter data={points} fill={cost ? blue : orange} /></ScatterChart></ResponsiveContainer></div></Panel>;
}

export function PredictionAccuracyPage() {
  const [selectedWindow, setSelectedWindow] = useState(() => {
    const stored = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY);
    return stored && SAVED_WINDOWS.some((item) => item.key === stored) ? stored : '2001_2017';
  });
  const [data, setData] = useState<PredictionAccuracyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const refresh = useCallback(() => setRefreshKey((value) => value + 1), []);
  useEffect(() => {
    const updateSelectedWindow = () => {
      const stored = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY);
      if (stored && SAVED_WINDOWS.some((item) => item.key === stored)) setSelectedWindow(stored);
    };
    globalThis.addEventListener('storage', updateSelectedWindow);
    return () => globalThis.removeEventListener('storage', updateSelectedWindow);
  }, []);
  useEffect(() => { const controller = new AbortController(); setLoading(true); setError(null); getPredictionAccuracyData(selectedWindow, controller.signal).then(setData).catch((reason: unknown) => { if (reason instanceof DOMException && reason.name === 'AbortError') return; setError(reason instanceof Error ? reason.message : 'Validation data unavailable.'); }).finally(() => { if (!controller.signal.aborted) setLoading(false); }); return () => controller.abort(); }, [refreshKey, selectedWindow]);
  const matrix = useMemo(() => { const result = Array.from({ length: 4 }, () => [0, 0, 0, 0]); for (const row of data?.rows ?? []) { const actual = typeof row.actual_risk === 'number' ? row.actual_risk : ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].indexOf(row.actual_risk.toUpperCase()); const predicted = typeof row.predicted_risk === 'number' ? row.predicted_risk : ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].indexOf(row.predicted_risk.toUpperCase()); if (actual >= 0 && actual < 4 && predicted >= 0 && predicted < 4) result[actual][predicted] += 1; } return result; }, [data]);
  if (!data && loading) return <div className="prediction-accuracy-page"><main className="pa-content"><Panel title="Prediction Accuracy">Loading real model validation artifacts…</Panel></main></div>;
  if (!data) return <div className="prediction-accuracy-page"><main className="pa-content"><Panel title="Prediction Accuracy"><p>{error ?? 'Validation data unavailable.'}</p><button className="pa-refresh" onClick={refresh}>Try again</button></Panel></main></div>;
  const { report } = data;
  const risk = report.risk_model;
  return <div className="prediction-accuracy-page">
    <div className="pa-product-header"><div><BarChart3 size={17} /><b>PAIMANA</b><span>MoSPI · Project Risk Intelligence</span></div><div><span className="pa-live"><i />Live · model {report.model_version}</span><em>SIH 26103</em></div></div>
    <main className="pa-content">
      <div className="pa-page-heading"><div><h1>Prediction Accuracy</h1><p>Real future-holdout evaluation for saved training range {selectedWindow.replace('_', '–')}.</p></div><div className="pa-heading-actions"><span className="pa-validated"><CheckCircle2 size={14} />Validated on {data.total} completed projects</span><span>Holdout: {report.metadata.test_start}–{report.metadata.test_end}</span><button className="pa-refresh" onClick={refresh} disabled={loading}><RefreshCw size={14} className={loading ? 'spin' : ''} />{loading ? 'Refreshing' : 'Refresh'}</button></div></div>
      {error && <p className="pa-note"><AlertTriangle size={13} />{error}</p>}
      <section className="pa-config"><div className="pa-status"><span>Model version</span><b>{report.model_version}</b></div><div className="pa-status"><span>Training period</span><b>{report.metadata.training_start}–{report.metadata.training_end}</b></div><div className="pa-status"><span>Holdout period</span><b>{report.metadata.test_start}–{report.metadata.test_end}</b></div><div className="pa-status"><span>Evaluation</span><b>{report.metadata.validation_method ?? 'Temporal holdout'}</b></div></section>
      <section className="pa-section"><h2>Validation Summary</h2><div className="pa-summary-grid">
        <SummaryCard title="Cost Prediction" icon={<TrendingUp size={16} />} footer="Lower MAE is better"><Value label="MAE" value={`${metricValue(report.cost_model.MAE)} pp`} /><Value label="MAPE" value={report.cost_model.MAPE === undefined ? 'Unavailable' : `${metricValue(report.cost_model.MAPE)}%`} /><Value label="R²" value={metricValue(report.cost_model.R2)} /></SummaryCard>
        <SummaryCard title="Delay Prediction" icon={<CalendarDays size={16} />} footer="Lower MAE is better"><Value label="MAE" value={`${metricValue(report.delay_model.MAE_days ?? report.delay_model.MAE)} days`} /><Value label="RMSE" value={report.delay_model.RMSE === undefined ? 'Unavailable' : `${metricValue(report.delay_model.RMSE)} days`} /><Value label="R²" value={metricValue(report.delay_model.R2)} /></SummaryCard>
        <SummaryCard title="Risk Classification" icon={<AlertTriangle size={16} />} accent="orange" footer="Higher is better"><Value label="F1" value={metricValue(risk.f1 ?? risk.macro_f1)} /><Value label="Precision" value={metricValue(risk.precision ?? risk.macro_precision)} /><Value label="Recall" value={metricValue(risk.recall ?? risk.macro_recall)} /></SummaryCard>
        <SummaryCard title="Temporal Stability" icon={<Clock3 size={16} />} footer={data.rolling.policy ?? 'Expanding-window validation'}><Value label="Windows" value={String(data.rolling.fold_count)} /><Value label="Evidence rows" value={String(data.total)} /></SummaryCard>
      </div></section>
      <section className="pa-analytics-row-one"><ScatterPanel title="Predicted vs Actual Cost Overrun (%)" rows={data.rows} kind="cost" /><ScatterPanel title="Predicted vs Actual Delay (Days)" rows={data.rows} kind="delay" /><Panel title="Risk Classification – Confusion Matrix" className="pa-matrix"><div className="pa-matrix-wrap"><table><thead><tr><th>Actual / Predicted</th>{riskNames.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{matrix.map((values, index) => <tr key={riskNames[index]}><th>{riskNames[index]}</th>{values.map((value, column) => <td key={column} className={index === column ? 'correct' : ''}>{value}</td>)}</tr>)}</tbody></table></div><footer>Accuracy: <b>{risk.accuracy.toFixed(3)}</b></footer></Panel></section>
      <section className="pa-analytics-row-two"><Panel title="Expanding Window Validation (MAE)" className="pa-window"><div className="pa-chart"><ResponsiveContainer><LineChart data={data.rolling.folds}><CartesianGrid stroke="#e3eaf3" /><XAxis dataKey="test_year" tick={{ fontSize: 9 }} /><YAxis yAxisId="cost" tick={{ fontSize: 9 }} /><YAxis yAxisId="delay" orientation="right" tick={{ fontSize: 9 }} /><Tooltip /><Legend /><Line yAxisId="cost" dataKey="cost_MAE" name="Cost MAE" stroke={blue} dot={false} /><Line yAxisId="delay" dataKey="delay_MAE_days" name="Delay MAE" stroke={orange} dot={false} /></LineChart></ResponsiveContainer></div></Panel><Panel title="Production Cost Feature Importance" className="pa-shap">{data.importance ? <div className="pa-chart"><ResponsiveContainer><BarChart data={data.importance.cost_model.slice(0, 5)} layout="vertical" margin={{ left: 24 }}><CartesianGrid stroke="#e3eaf3" horizontal={false} /><XAxis type="number" tick={{ fontSize: 9 }} /><YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 9 }} /><Tooltip /><Bar dataKey="importance" fill={blue} /></BarChart></ResponsiveContainer></div> : <p className="section-unavailable">Feature-importance artifacts were not published for this saved evaluation.</p>}</Panel></section>
      <Panel title="Project-wise Prediction Evidence" className="pa-evidence"><div className="pa-table-wrap"><table><thead><tr>{['Project ID','Project Name','Sector','Actual Cost Overrun (%)','Predicted Cost Overrun (%)','Cost Error (pp)','Actual Delay (Days)','Predicted Delay (Days)','Delay Error (Days)','Actual Risk','Predicted Risk','Confidence'].map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{data.rows.slice(0, 20).map((row, index) => <tr key={`${row.project_id ?? 'unreported'}-${index}`}><td className="mono">{row.project_id ?? 'Not reported'}</td><td><b>{row.project_name}</b></td><td>{row.sector}</td><td>{row.actual_cost_overrun.toFixed(1)}</td><td>{row.predicted_cost_overrun.toFixed(1)}</td><td>{row.cost_error.toFixed(1)}</td><td>{row.actual_delay_days.toFixed(0)}</td><td>{row.predicted_delay_days.toFixed(0)}</td><td>{row.delay_error.toFixed(0)}</td><td>{riskLabel(row.actual_risk)}</td><td>{riskLabel(row.predicted_risk)}</td><td>{row.model_confidence_percentage === null ? 'Unavailable' : `${row.model_confidence_percentage.toFixed(1)}%`}</td></tr>)}</tbody></table></div></Panel>
      <p className="pa-note"><Info size={13} />{report.metadata.data_source ?? 'Official production validation artifacts'} · metrics are not recomputed in the browser.</p>
    </main>
  </div>;
}
