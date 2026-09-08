import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { BarChart3, CalendarDays, FileText, RefreshCw, TableProperties, TrendingDown } from 'lucide-react';
import { CartesianGrid, LabelList, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getTrainingWindowPerformance, type TrainingWindowMetric, type TrainingWindowPerformance } from '../services/trainingWindowPerformanceService';
import '../styles/trainingWindowPerformance.css';

const formatWindow = (item: Pick<TrainingWindowMetric, 'start_year' | 'end_year'>) => `${item.start_year} – ${item.end_year}`;
const metric = (value: number | null) => value === null || !Number.isFinite(value) ? 'Unavailable' : value.toFixed(3);
const reduction = (baseline: number | null, value: number | null) => baseline === null || value === null || baseline === 0 ? null : ((baseline - value) / baseline) * 100;
const dateLabel = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
const status = (index: number) => index === 0 ? 'Weakest' : index === 1 ? 'Improved' : 'Best';

function Panel({ title, icon, children, className = '', subtitle }: { title: string; icon: ReactNode; children: ReactNode; className?: string; subtitle?: string }) {
  return <section className={`twp-panel ${className}`}><header><span className="twp-panel-icon">{icon}</span><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div></header>{children}</section>;
}

function WindowCard({ item, index }: { item: TrainingWindowMetric; index: number }) {
  const badge = status(index);
  return <article className="twp-window-card"><div className="twp-window-head"><span className="twp-calendar"><CalendarDays size={19} /></span><div><span>Training Window</span><h3>{formatWindow(item)}</h3><p>Testing: 2023 – 2025</p></div><b className={`twp-status ${badge.toLowerCase()}`}>{badge}</b></div><div className="twp-card-metrics"><div><span>Cost MAE</span><strong>{metric(item.cost_mae)}<small>%</small></strong></div><div><span>Delay MAE</span><strong>{metric(item.delay_mae_days)}<small> days</small></strong></div></div></article>;
}

function ErrorTrendCard({ windows }: { windows: TrainingWindowMetric[] }) {
  const chart = (key: 'cost_mae' | 'delay_mae_days', label: string, color: string, suffix: string) => <div className="twp-trend" key={key}><div className="twp-trend-title"><h3>{label}</h3><span><i style={{ backgroundColor: color }} />{suffix}</span></div><div className="twp-chart"><ResponsiveContainer><LineChart data={windows} margin={{ top: 25, right: 18, left: -12, bottom: 0 }}><CartesianGrid stroke="#e4eaf2" /><XAxis dataKey="end_year" tick={{ fontSize: 12, fill: '#64748b' }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} /><YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} width={48} domain={['auto', 'auto']} /><Tooltip formatter={(value) => [typeof value === 'number' ? `${value.toFixed(3)}${key === 'cost_mae' ? '%' : ' days'}` : 'Unavailable', suffix]} labelFormatter={(year) => `Training window ending ${String(year)}`} /><Line type="monotone" dataKey={key} name={suffix} stroke={color} strokeWidth={3} dot={{ r: 5, fill: color, stroke: '#fff', strokeWidth: 2 }} activeDot={{ r: 6 }}><LabelList dataKey={key} position="top" formatter={(value) => typeof value === 'number' ? value.toFixed(3) : '—'} style={{ fill: '#1e293b', fontSize: 11, fontWeight: 800 }} /></Line></LineChart></ResponsiveContainer></div></div>;
  return <Panel title="Cost and Delay Error Trend" icon={<BarChart3 size={18} />} className="twp-error-panel">{chart('cost_mae', 'Cost MAE (Lower is Better)', '#1769c2', 'Cost MAE')}{chart('delay_mae_days', 'Delay MAE (Lower is Better)', '#2fa8e0', 'Delay MAE')}</Panel>;
}

function ImprovementCard({ windows }: { windows: TrainingWindowMetric[] }) {
  const baseline = windows[0];
  const rows = windows.slice(1);
  const renderRows = (key: 'cost_mae' | 'delay_mae_days') => rows.map((item, index) => {
    const value = reduction(baseline?.[key] ?? null, item[key]);
    const width = value === null ? 0 : Math.min(100, Math.max(0, value * 2));
    return <div className="twp-improvement-row" key={`${key}-${item.end_year}`}><span>{formatWindow(item)}</span><div className="twp-bar-track"><i className={index === 1 ? 'best' : ''} style={{ width: `${width}%` }} /></div><b>{value === null ? 'Unavailable' : `${value.toFixed(1)}%`}</b></div>;
  });
  const axis = <div className="twp-axis"><span>0</span><span>10</span><span>20</span><span>30</span><span>40</span><span>50</span></div>;
  return <Panel title="Relative Improvement vs 2001 – 2017" subtitle="More recent data, better calibration." icon={<TrendingDown size={18} />} className="twp-improvement-panel"><div className="twp-improvement-section"><h3>Cost Error Reduction</h3><div className="twp-improvement-plot">{renderRows('cost_mae')}{axis}<label>Error reduction (%)</label></div><aside>More recent data,<br />better calibration.</aside></div><div className="twp-improvement-section"><h3>Delay Error Reduction</h3><div className="twp-improvement-plot">{renderRows('delay_mae_days')}{axis}<label>Error reduction (%)</label></div><aside>Newer projects improve delay prediction accuracy.</aside></div></Panel>;
}

function InterpretationCard() {
  const items = ['More recent data captures current execution patterns.', 'Models trained on newer windows calibrate better to present-day projects.', 'Both cost and delay errors decline as the window end-year moves forward.', 'This supports the SIH proof-of-concept case for continuous data expansion.'];
  return <Panel title="Interpretation" icon={<FileText size={18} />} className="twp-interpretation-panel"><ol>{items.map((item, index) => <li key={item}><b>{index + 1}</b><span>{item}</span></li>)}</ol></Panel>;
}

function QuickComparison({ windows }: { windows: TrainingWindowMetric[] }) {
  return <Panel title="Quick Comparison" icon={<TableProperties size={18} />} className="twp-table-panel"><div className="twp-table-wrap"><table><thead><tr><th>Train → Test</th><th>Cost MAE ↓</th><th>Delay MAE ↓</th><th>R² ↑</th></tr></thead><tbody>{windows.map((item, index) => <tr className={index === 2 ? 'best-row' : ''} key={item.end_year}><td>{item.start_year}–{item.end_year} → 2023–2025</td><td>{metric(item.cost_mae)}%</td><td>{metric(item.delay_mae_days)} days</td><td>{item.cost_r2 === null ? '—' : item.cost_r2.toFixed(2)}</td></tr>)}</tbody></table></div></Panel>;
}

export function TrainingWindowPerformancePage() {
  const [data, setData] = useState<TrainingWindowPerformance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const retry = useCallback(() => setRefresh((value) => value + 1), []);
  useEffect(() => { const controller = new AbortController(); setLoading(true); setError(null); getTrainingWindowPerformance(controller.signal).then((result) => setData({ ...result, windows: [...result.windows].sort((a, b) => a.end_year - b.end_year) })).catch((reason: unknown) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Training-window metrics could not be loaded.'); }).finally(() => { if (!controller.signal.aborted) setLoading(false); }); return () => controller.abort(); }, [refresh]);
  if (loading && !data) return <main className="twp-page"><div className="twp-heading skeleton"><div /><span /></div><section className="twp-summary-grid">{[0, 1, 2].map((item) => <div className="twp-skeleton-card" key={item} />)}</section></main>;
  if (!data) return <main className="twp-page"><section className="twp-error"><h1>Training Window Performance</h1><p>{error ?? 'Training-window metrics could not be loaded.'}</p><button onClick={retry}><RefreshCw size={15} />Retry</button></section></main>;
  return <main className="twp-page"><header className="twp-heading"><div><h1>Training Window Performance</h1><p>Why newer data improves the proof of concept</p></div><div><span>Last updated</span><b>{dateLabel(data.generated_at)}</b></div></header><section className="twp-summary-grid">{data.windows.map((item, index) => <WindowCard key={item.end_year} item={item} index={index} />)}</section><section className="twp-main-grid"><ErrorTrendCard windows={data.windows} /><ImprovementCard windows={data.windows} /></section><section className="twp-bottom-grid"><InterpretationCard /><QuickComparison windows={data.windows} /></section><p className="twp-methodology">{data.methodology}</p></main>;
}
