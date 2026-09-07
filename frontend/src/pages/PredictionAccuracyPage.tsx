import { type ReactNode, useState } from 'react';
import { AlertTriangle, BarChart3, CalendarDays, CheckCircle2, ClipboardList, Clock3, Layers3, ShieldCheck, TrendingUp } from 'lucide-react';
import '../styles/predictionAccuracy.css';

const EVALUATED = {
  trainingProjects: 1578,
  costMae: 23.750,
  costR2: 0.3796,
  delayMae: 343.592,
  delayR2: 0.7716,
} as const;

function Panel({ title, icon, children, className = '', action }: { title: string; icon: ReactNode; children: ReactNode; className?: string; action?: ReactNode }) {
  return <section className={`pa-panel ${className}`}><header><h2>{icon}{title}</h2>{action}</header>{children}</section>;
}

function Empty() { return <p className="pa-empty" aria-label="No evaluated data available" />; }

function Kpi({ title, icon, tone, result, footer }: { title: string; icon: ReactNode; tone: string; result: string; footer: string }) {
  return <article className="pa-kpi"><span className={tone}>{icon}</span><div><h3>{title}</h3><strong>{result}</strong><p>{footer}</p></div></article>;
}

function MiniMetric({ label, result, tone }: { label: string; result: string; tone: string }) {
  return <div className={`pa-mini ${tone}`}><span>{label}</span><b>{result}</b></div>;
}

export function PredictionAccuracyPage() {
  const [more, setMore] = useState(false);
  const facts: Array<[typeof CalendarDays, string, string]> = [
    [CalendarDays, 'Training period', '2001 – 2021'],
    [Layers3, 'Training projects', EVALUATED.trainingProjects.toLocaleString()],
  ];

  return <div className="prediction-accuracy-page">
    <div className="pa-product-header"><div><b>PAIMANA</b><span>MoSPI · Project Risk Intelligence</span></div><div><em className="production"><i />Production model</em><em>SIH 26103</em></div></div>
    <main className="pa-content">
      <div className="pa-heading"><div><h1>Prediction Accuracy</h1><p>Training period · 2001–2021</p></div><span className="verified"><CheckCircle2 />Verified production evidence</span></div>
      <section className="pa-evidence" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>{facts.map(([Icon, label, result], index) => <div key={label}><Icon /><span>{label}</span><b>{result}</b>{index < facts.length - 1 && <i />}</div>)}</section>
      <section className="pa-kpis">
        <Kpi icon={<BarChart3 />} tone="blue" title="Cost Forecast" result={`${EVALUATED.costMae.toFixed(3)} pp`} footer="MAE" />
        <Kpi icon={<Clock3 />} tone="orange" title="Delay Forecast" result={`${EVALUATED.delayMae.toFixed(3)} days`} footer="MAE" />
        <Kpi icon={<AlertTriangle />} tone="red" title="Risk Classification" result="" footer="" />
        <Kpi icon={<ShieldCheck />} tone="green" title="Training Window" result="2001–2021" footer="" />
      </section>
      <section className="pa-main">
        <Panel title="Predicted vs Actual Cost" icon={<TrendingUp />} className="pa-chart-panel"><Empty /></Panel>
        <Panel title="Predicted vs Actual Delay" icon={<Clock3 />} className="pa-chart-panel"><Empty /></Panel>
        <Panel title="Error Summary" icon={<ClipboardList />}><Empty /></Panel>
      </section>
      <section className="pa-lower">
        <Panel title="Regression Metrics" icon={<BarChart3 />}><div className="regression"><div className="model cost"><h3>Cost model</h3><p><span>MAE</span><b>{EVALUATED.costMae.toFixed(3)} pp</b></p><p><span>R²</span><b>{EVALUATED.costR2.toFixed(4)}</b></p><p><span>MAPE</span><b /></p></div><div className="model delay"><h3>Delay model</h3><p><span>MAE</span><b>{EVALUATED.delayMae.toFixed(3)} d</b></p><p><span>R²</span><b>{EVALUATED.delayR2.toFixed(4)}</b></p><p><span>MAPE</span><b /></p></div></div></Panel>
        <Panel title="Classification Metrics" icon={<TrendingUp />}><div className="classification"><MiniMetric label="Precision" result="" tone="blue" /><MiniMetric label="Recall" result="" tone="orange" /><MiniMetric label="F1" result="" tone="green" /></div></Panel>
        <Panel title="Performance Across Time" icon={<Clock3 />}><Empty /></Panel>
      </section>
      <Panel title="Project-level Validation Sample" icon={<ClipboardList />} className="sample" action={<button className="more" onClick={() => setMore(value => !value)}>{more ? 'Show fewer samples' : 'View more samples'} →</button>}><div className="table-wrap"><table><thead><tr>{['Project ID', 'Predicted Cost (%)', 'Actual Cost (%)', 'Cost Error (pp)', 'Predicted Delay (days)', 'Actual Delay (days)', 'Delay Error (days)', 'Confidence'].map(label => <th key={label}>{label}</th>)}</tr></thead><tbody /></table></div></Panel>
      <p className="footer-note">Metrics reflect the evaluated 2001–2021 training window.</p>
    </main>
  </div>;
}
