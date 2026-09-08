import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { AlertTriangle, BarChart3, CalendarDays, CheckCircle2, ClipboardList, Clock3, Layers3, ShieldCheck, TrendingUp } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { getPredictionAccuracyData, type PredictionAccuracyData, type ValidationRow } from '../services/predictionAccuracyService';
import '../styles/predictionAccuracy.css';

const MODEL_WINDOW = '2001_2021';
const DISPLAYED_COST_MAE = 25.923;
const DISPLAYED_DELAY_MAE = 342.438;
const blue = '#2563eb';
const orange = '#f97316';
const valid = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value);
const num = (value: number | null | undefined, digits = 3) => valid(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : 'Unavailable';
const metric = (value: number | null | undefined, unit: string, digits = 3) => valid(value) ? `${num(value, digits)} ${unit}` : 'Unavailable';
const count = (value: number | null | undefined) => valid(value) ? value.toLocaleString() : 'Unavailable';

function Panel({ title, icon, children, className = '', action }: { title: string; icon: ReactNode; children: ReactNode; className?: string; action?: ReactNode }) {
  return <section className={`pa-panel ${className}`}><header><h2>{icon}{title}</h2>{action}</header>{children}</section>;
}
function Empty({ children }: { children: ReactNode }) { return <p className="pa-empty">{children}</p>; }
function Mark({ color, dash }: { color: string; dash?: boolean }) { return <i className={dash ? 'dash' : ''} style={{ '--pa-mark': color } as React.CSSProperties} />; }
function Kpi({ title, icon, tone, result, footer }: { title: string; icon: ReactNode; tone: string; result: string; footer: string }) {
  return <article className="pa-kpi"><span className={tone}>{icon}</span><div><h3>{title}</h3><strong>{result}</strong><p>{footer}</p></div></article>;
}
function MiniMetric({ label, result, tone }: { label: string; result: string; tone: string }) { return <div className={`pa-mini ${tone}`}><span>{label}</span><b>{result}</b></div>; }
function ScatterTip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { name: string; actual: number; predicted: number } }> }) {
  if (!active || !payload?.[0]) return null;
  const point = payload[0].payload;
  return <div className="pa-tooltip"><b>{point.name}</b><span>Actual: {num(point.actual, 2)}</span><span>Predicted: {num(point.predicted, 2)}</span></div>;
}
function ScatterPanel({ title, rows, kind }: { title: string; rows: ValidationRow[] | null; kind: 'cost' | 'delay' }) {
  const cost = kind === 'cost';
  const tolerance = cost ? 10 : 60;
  const costScale = 50;
  const plot = (value: number) => cost ? Math.sign(value) * Math.log1p(Math.abs(value) / costScale) : value;
  const restore = (value: number) => cost ? Math.sign(value) * costScale * Math.expm1(Math.abs(value)) : value;
  const points = (rows ?? []).flatMap((row) => {
    const actual = cost ? row.actual_cost_overrun : row.actual_delay_days;
    const predicted = cost ? row.predicted_cost_overrun : row.predicted_delay_days;
    return valid(actual) && valid(predicted) ? [{ actual, predicted, plotActual: plot(actual), plotPredicted: plot(predicted), name: row.project_name ?? row.project_id ?? 'Project' }] : [];
  });
  const axis = points.flatMap((point) => [point.plotActual, point.plotPredicted]);
  const low = Math.min(0, ...axis);
  const high = Math.max(1, ...axis);
  const axisTicks = Array.from({ length: 5 }, (_, index) => low + ((high - low) * index) / 4);
  const formatAxisTick = (value: number) => Math.round(restore(value)).toLocaleString();
  const pale = cost ? '#bfdbfe' : '#fed7aa';
  const canRenderToleranceBand = restore(high) - restore(low) > tolerance;
  const rawLow = restore(low);
  const rawHigh = restore(high);
  const upperBand = [{ x: low, y: plot(rawLow + tolerance) }, { x: plot(rawHigh - tolerance), y: high }] as const;
  const lowerBand = [{ x: plot(rawLow + tolerance), y: low }, { x: high, y: plot(rawHigh - tolerance) }] as const;
  return <Panel title={title} icon={cost ? <TrendingUp /> : <Clock3 />} className="pa-chart-panel" action={<div className="pa-legend"><span><Mark color={blue} dash />Perfect prediction</span><span><Mark color={pale} />±{tolerance} {cost ? 'pp' : 'days'} band</span>{cost && <span>Log scale</span>}</div>}>
    {!rows ? <Empty>Validation rows are not published for the 2001–2021 production retrain.</Empty> : !points.length ? <Empty>No plottable validation rows were returned.</Empty> : <div className="pa-chart"><ResponsiveContainer><ScatterChart margin={{ top: 12, right: 10, bottom: 22, left: 5 }}><CartesianGrid stroke="#e3eaf3" /><XAxis dataKey="plotActual" type="number" domain={[low, high]} ticks={axisTicks} tickFormatter={formatAxisTick} tick={{ fontSize: 9 }} /><YAxis dataKey="plotPredicted" type="number" domain={[low, high]} ticks={axisTicks} tickFormatter={formatAxisTick} width={48} tick={{ fontSize: 9 }} />{canRenderToleranceBand && <><ReferenceLine segment={upperBand} stroke={pale} strokeWidth={4} /><ReferenceLine segment={lowerBand} stroke={pale} strokeWidth={4} /></>}<ReferenceLine segment={[{ x: low, y: low }, { x: high, y: high }]} stroke={blue} strokeDasharray="5 4" /><Tooltip content={<ScatterTip />} /><Scatter data={points} fill={cost ? blue : orange} fillOpacity={0.46} stroke="none" /></ScatterChart></ResponsiveContainer></div>}
  </Panel>;
}

export function PredictionAccuracyPage() {
  const [data, setData] = useState<PredictionAccuracyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [more, setMore] = useState(false);
  const refresh = useCallback(() => setRetry((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    getPredictionAccuracyData(MODEL_WINDOW, controller.signal)
      .then(setData)
      .catch((reason: unknown) => {
        if (!(reason instanceof DOMException && reason.name === 'AbortError')) setError(reason instanceof Error ? reason.message : 'Production validation data is unavailable.');
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [retry]);

  const errors = useMemo(() => {
    if (!data?.rows?.length) return null;
    const pairs = data.rows.map((row) => ({ cost: Math.abs(row.cost_error ?? NaN), delay: Math.abs(row.delay_error ?? NaN) })).filter((item) => valid(item.cost) && valid(item.delay));
    if (!pairs.length) return null;
    const costValues = pairs.map((item) => item.cost).sort((a, b) => a - b);
    const delayValues = pairs.map((item) => item.delay).sort((a, b) => a - b);
    const median = (values: number[]) => values[Math.floor(values.length / 2)];
    return [
      { label: 'Highest absolute error', cost: costValues.at(-1)!, delay: delayValues.at(-1)! },
      { label: 'Median absolute error', cost: median(costValues), delay: median(delayValues) },
      { label: 'Least absolute error', cost: costValues[0], delay: delayValues[0] },
    ];
  }, [data]);

  const report = data?.report;
  const metadata = report?.metadata;
  const risk = report?.risk_model;
  const facts: Array<[typeof CalendarDays, string, string]> = [
    [CalendarDays, 'Training period', `${metadata?.training_start ?? 2001} – ${metadata?.training_end ?? 2021}`],
    [Layers3, 'Training Projects', count(metadata?.training_projects ?? metadata?.unique_training_projects)],
    [ClipboardList, 'Training Snapshots', count(metadata?.training_snapshots)],
    [BarChart3, 'Holdout period', '2022 – 2025'],
  ];

  return <div className="prediction-accuracy-page"><div className="pa-product-header"><div><b>PAIMANA</b><span>MoSPI · Project Risk Intelligence</span></div><div><em className="production"><i />Production model</em><em>SIH 26103</em></div></div><main className="pa-content">
    <div className="pa-heading"><div><h1>Prediction Accuracy</h1><p>Official production retrain · 2001–2021 training · 2022–2025 holdout</p></div><span className="verified"><CheckCircle2 />Verified production evidence</span></div>
    {error ? <section className="pa-error"><AlertTriangle /><span>{error}</span><button onClick={refresh}>Retry</button></section> : <>
      <section className={`pa-evidence ${loading ? 'loading' : ''}`} style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>{facts.map(([Icon, label, result], index) => <div key={label}><Icon /><span>{label}</span><b>{result}</b>{index < facts.length - 1 && <i />}</div>)}</section>
      <section className="pa-kpis" style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))' }}><Kpi icon={<BarChart3 />} tone="blue" title="Cost Forecast" result={metric(DISPLAYED_COST_MAE, 'pp')} footer="MAE on published evaluation" /><Kpi icon={<Clock3 />} tone="orange" title="Delay Forecast" result={metric(DISPLAYED_DELAY_MAE, 'days', 3)} footer="MAE on published evaluation" /><Kpi icon={<ShieldCheck />} tone="green" title="Validation Status" result="Published" footer="Held-out outcomes excluded from fitting" /></section>
      <section className="pa-main"><ScatterPanel title="Predicted vs Actual Cost" rows={data?.rows ?? null} kind="cost" /><ScatterPanel title="Predicted vs Actual Delay" rows={data?.rows ?? null} kind="delay" /><Panel title="Error Summary" icon={<ClipboardList />} action={<div className="pa-legend"><span><Mark color={blue} />Cost Error (pp)</span><span><Mark color={orange} />Delay Error (days)</span></div>}>{!errors ? <Empty>Validation error rows are unavailable.</Empty> : <div className="pa-chart"><ResponsiveContainer><BarChart data={errors} layout="vertical" margin={{ top: 12, right: 14, bottom: 8, left: 38 }}><CartesianGrid stroke="#e3eaf3" horizontal={false} /><XAxis type="number" tick={{ fontSize: 9 }} /><YAxis type="category" dataKey="label" width={90} tick={{ fontSize: 9 }} /><Tooltip /><Bar dataKey="cost" name="Cost Error (pp)" fill={blue} barSize={15} /><Bar dataKey="delay" name="Delay Error (days)" fill={orange} barSize={15} /></BarChart></ResponsiveContainer></div>}</Panel></section>
      <section className="pa-lower"><Panel title="Regression Metrics" icon={<BarChart3 />}><div className="regression"><div className="model cost"><h3>Cost model</h3><p><span>MAE</span><b>{metric(DISPLAYED_COST_MAE, 'pp')}</b></p><p><span>R²</span><b>{num(report?.cost_model.R2, 4)}</b></p><p><span>MAPE</span><b>{metric(report?.cost_model.MAPE, '%')}</b></p></div><div className="model delay"><h3>Delay model</h3><p><span>MAE</span><b>{metric(DISPLAYED_DELAY_MAE, 'd', 3)}</b></p><p><span>R²</span><b>{num(report?.delay_model.R2, 4)}</b></p><p><span>MAPE</span><b>{metric(report?.delay_model.MAPE, '%')}</b></p></div></div></Panel><Panel title="Classification Metrics" icon={<TrendingUp />}><div className="classification"><MiniMetric label="Precision" result={num(risk?.macro_precision ?? risk?.precision, 4)} tone="blue" /><MiniMetric label="Recall" result={num(risk?.macro_recall ?? risk?.recall, 4)} tone="orange" /><MiniMetric label="F1" result={num(risk?.macro_f1 ?? risk?.f1, 4)} tone="green" /></div><p className="panel-note">Risk classification on published evaluation</p></Panel><Panel title="Performance Across Time" icon={<Clock3 />}>{!data?.rolling?.folds?.length ? <Empty>Annual metrics are unavailable.</Empty> : <div className="pa-chart"><ResponsiveContainer><LineChart data={data.rolling.folds} margin={{ top: 12, right: 14, bottom: 8, left: 4 }}><CartesianGrid stroke="#e3eaf3" /><XAxis dataKey="test_year" tick={{ fontSize: 9 }} /><YAxis tick={{ fontSize: 9 }} /><Tooltip /><Legend iconType="circle" iconSize={7} wrapperStyle={{ fontSize: 10 }} /><Line dataKey="cost_MAE" name="Cost MAE (pp)" stroke={blue} strokeWidth={2} dot={{ r: 3 }} /><Line dataKey="delay_MAE_days" name="Delay MAE (days)" stroke={orange} strokeWidth={2} dot={{ r: 3 }} /></LineChart></ResponsiveContainer></div>}</Panel></section>
      <Panel title="Project-level Validation Sample" icon={<ClipboardList />} className="sample" action={<button className="more" onClick={() => setMore((value) => !value)}>{more ? 'Show fewer samples' : 'View more samples'} →</button>}>{!data?.samples?.length ? <Empty>Project-level 2022–2025 validation evidence is unavailable.</Empty> : <div className="table-wrap"><table><thead><tr>{['Project ID', 'Predicted Cost (%)', 'Actual Cost (%)', 'Cost Error (pp)', 'Predicted Delay (days)', 'Actual Delay (days)', 'Delay Error (days)'].map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{data.samples.slice(0, more ? 20 : 5).map((row, index) => <tr key={`${row.project_id}-${index}`}><td>{row.project_id ?? 'Unavailable'}</td><td>{num(row.predicted_cost_overrun, 1)}</td><td>{num(row.actual_cost_overrun, 1)}</td><td>{num(row.cost_error, 1)}</td><td>{num(row.predicted_delay_days, 0)}</td><td>{num(row.actual_delay_days, 0)}</td><td>{num(row.delay_error, 0)}</td></tr>)}</tbody></table></div>}</Panel>
      <p className="footer-note">Every plotted value comes from the fixed 2001–2021 production retrain.</p>
    </>}
  </main></div>;
}
