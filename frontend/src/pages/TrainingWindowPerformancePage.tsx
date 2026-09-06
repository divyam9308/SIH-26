import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { BarChart3, CalendarDays, FileText, RefreshCw, TableProperties, TrendingDown } from 'lucide-react';
import { CartesianGrid, LabelList, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getTrainingWindowPerformance, type TrainingWindowMetric, type TrainingWindowPerformance } from '../services/trainingWindowPerformanceService';
import '../styles/trainingWindowPerformance.css';

const formatWindow = (item: Pick<TrainingWindowMetric, 'start_year' | 'end_year'>) => `${item.start_year} – ${item.end_year}`;
const number = (value: number | null, digits: number) => value === null || !Number.isFinite(value) ? 'Unavailable' : value.toFixed(digits);
const reduction = (baseline: number | null, value: number | null) => baseline === null || value === null || baseline === 0 ? null : ((baseline - value) / baseline) * 100;
const dateLabel = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));

function Panel({ title, icon, children, className = '' }: { title: string; icon: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`twp-panel ${className}`}><header><span className="twp-panel-icon">{icon}</span><h2>{title}</h2></header>{children}</section>;
}

function WindowCard({ item }: { item: TrainingWindowMetric }) {
  return <article className="twp-window-card"><div className="twp-window-head"><span className="twp-calendar"><CalendarDays size={18} /></span><div><span>Training Window</span><h3>{formatWindow(item)}</h3></div><b className="twp-status improved">Verified</b></div><div className="twp-card-metrics"><div><span>Cost MAE</span><strong>{number(item.cost_mae, 1)}<small> pp</small></strong></div><div><span>Delay MAE</span><strong>{number(item.delay_mae_days, 1)}<small> days</small></strong></div></div></article>;
}

function ErrorTrendCard({ windows }: { windows: TrainingWindowMetric[] }) {
  const chart = (key: 'cost_mae' | 'delay_mae_days', label: string, color: string, suffix: string) => <div className="twp-trend" key={key}><div className="twp-trend-title"><h3>{label}</h3><span><i style={{ backgroundColor: color }} />{suffix}</span></div><div className="twp-chart"><ResponsiveContainer><LineChart data={windows} margin={{ top: 22, right: 16, left: -20, bottom: 0 }}><CartesianGrid vertical={false} stroke="#e8eef5" /><XAxis dataKey="end_year" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} /><YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} width={42} /><Tooltip formatter={(value) => [typeof value === 'number' ? `${value.toFixed(3)} ${suffix === 'Cost MAE' ? 'pp' : 'days'}` : 'Unavailable', suffix]} labelFormatter={(year) => `Window ending ${String(year)}`} /><Line type="monotone" dataKey={key} name={suffix} stroke={color} strokeWidth={2.5} dot={{ r: 4, fill: '#fff', stroke: color, strokeWidth: 2 }} activeDot={{ r: 5 }}><LabelList dataKey={key} position="top" formatter={(value) => typeof value === 'number' ? value.toFixed(1) : '—'} style={{ fill: '#334155', fontSize: 10, fontWeight: 700 }} /></Line></LineChart></ResponsiveContainer></div></div>;
  return <Panel title="Cost and Delay Error Trend" icon={<BarChart3 size={17} />} className="twp-error-panel">{chart('cost_mae', 'Cost MAE (Lower is Better)', '#2563eb', 'Cost MAE')}{chart('delay_mae_days', 'Delay MAE (Lower is Better)', '#38bdf8', 'Delay MAE')}</Panel>;
}

function ImprovementCard({ windows }: { windows: TrainingWindowMetric[] }) {
  const baseline = windows[0];
  const rows = windows.slice(1);
  const renderRows = (key: 'cost_mae' | 'delay_mae_days', color: string) => rows.map((item) => {
    const value = reduction(baseline?.[key] ?? null, item[key]);
    const width = value === null ? 0 : Math.min(100, Math.abs(value));
    return <div className="twp-improvement-row" key={`${key}-${item.end_year}`}><span>{formatWindow(item)}</span><div className="twp-bar-track"><i className={value !== null && value < 0 ? 'negative' : ''} style={{ width: `${width}%`, backgroundColor: value !== null && value < 0 ? '#fca5a5' : color }} /></div><b className={value !== null && value < 0 ? 'negative' : ''}>{value === null ? 'Unavailable' : `${value >= 0 ? '' : '−'}${Math.abs(value).toFixed(1)}%`}</b></div>;
  });
  const costSupport = rows.every((item) => (reduction(baseline?.cost_mae ?? null, item.cost_mae) ?? 0) > 0);
  const delaySupport = rows.every((item) => (reduction(baseline?.delay_mae_days ?? null, item.delay_mae_days) ?? 0) > 0);
  return <Panel title="Relative Change vs 2001 – 2020" icon={<TrendingDown size={17} />} className="twp-improvement-panel"><p className="twp-panel-subtitle">Each model was evaluated on its own later temporal holdout.</p><div className="twp-improvement-section"><h3>Cost MAE Change</h3>{renderRows('cost_mae', '#2563eb')}<aside>{costSupport ? 'Lower values are observed, but cohorts differ.' : 'Values are descriptive; cohorts differ.'}</aside></div><div className="twp-improvement-section"><h3>Delay MAE Change</h3>{renderRows('delay_mae_days', '#38bdf8')}<aside>{delaySupport ? 'Lower values are observed, but cohorts differ.' : 'Values are descriptive; cohorts differ.'}</aside></div></Panel>;
}

function InterpretationCard({ windows }: { windows: TrainingWindowMetric[] }) {
  const costDeclines = windows.every((item, index) => index === 0 || (item.cost_mae ?? Infinity) < (windows[index - 1].cost_mae ?? -Infinity));
  const delayDeclines = windows.every((item, index) => index === 0 || (item.delay_mae_days ?? Infinity) < (windows[index - 1].delay_mae_days ?? -Infinity));
  const items = [
    'Only provenance-verified canonical lifecycle evaluation artifacts are shown.',
    'Each window uses its own future temporal holdout; this is not a shared-cohort comparison.',
    costDeclines && delayDeclines ? 'Both displayed MAEs decline across these saved evaluations, but that does not establish a causal improvement.' : 'Observed differences reflect the saved temporal holdouts.',
    'A common holdout evaluation is required before ranking models or making a promotion decision.',
  ];
  return <Panel title="Interpretation" icon={<FileText size={17} />} className="twp-interpretation-panel"><ol>{items.map((item, index) => <li key={item}><b>{index + 1}</b><span>{item}</span></li>)}</ol></Panel>;
}

function QuickComparison({ windows }: { windows: TrainingWindowMetric[] }) {
  return <Panel title="Saved Evaluations" icon={<TableProperties size={17} />} className="twp-table-panel"><div className="twp-table-wrap"><table><thead><tr><th>Window</th><th>Cost MAE ↓</th><th>Delay MAE ↓</th><th>Cost R² ↑</th></tr></thead><tbody>{windows.map((item) => <tr key={item.end_year}><td>{formatWindow(item)}</td><td>{number(item.cost_mae, 1)} pp</td><td>{number(item.delay_mae_days, 1)} days</td><td>{number(item.cost_r2, 2)}</td></tr>)}</tbody></table></div></Panel>;
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
  return <main className="twp-page"><header className="twp-heading"><div><h1>Training Window Performance</h1></div><div><span>Last updated</span><b>{dateLabel(data.generated_at)}</b></div></header><section className="twp-summary-grid">{data.windows.map((item) => <WindowCard key={item.end_year} item={item} />)}</section><section className="twp-main-grid"><ErrorTrendCard windows={data.windows} /><ImprovementCard windows={data.windows} /></section><section className="twp-bottom-grid"><InterpretationCard windows={data.windows} /><QuickComparison windows={data.windows} /></section><p className="twp-methodology">{data.methodology}</p></main>;
}
