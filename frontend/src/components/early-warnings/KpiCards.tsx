import { Activity, AlertTriangle, Bell, ShieldAlert, TrendingUp } from 'lucide-react';
import { Area, AreaChart, Bar, BarChart, ResponsiveContainer } from 'recharts';
import { cn } from '../../lib/earlyWarningsUtils';
import type { EarlyWarningKpi } from '../../types/earlyWarnings';

type Tone = 'info' | 'critical' | 'warn' | 'success';
const presentation: Record<string, { label: string; icon: typeof Bell; tone: Tone; chart: 'line' | 'bar' }> = {
  new: { label: 'New Escalations', icon: Bell, tone: 'info', chart: 'line' }, critical: { label: 'Critical This Month', icon: AlertTriangle, tone: 'critical', chart: 'line' },
  persistent: { label: 'Persistent High Risk', icon: ShieldAlert, tone: 'warn', chart: 'line' }, improved: { label: 'Improved Projects', icon: TrendingUp, tone: 'success', chart: 'line' },
  average_change: { label: 'Avg Risk Change', icon: Activity, tone: 'info', chart: 'bar' },
};
const text = { info: 'text-info', critical: 'text-critical', warn: 'text-warn', success: 'text-success' } as const;
const background = { info: 'bg-info-soft', critical: 'bg-critical-soft', warn: 'bg-warn-soft', success: 'bg-success-soft' } as const;
const stroke = { info: 'var(--info)', critical: 'var(--critical)', warn: 'var(--warn)', success: 'var(--success)' } as const;

export function KpiCards({ kpis = [] }: { kpis?: EarlyWarningKpi[] }) { return <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">{kpis.map((kpi) => { const item = presentation[kpi.key]; if (!item) return null; const Icon = item.icon; const data = kpi.data.map((value, index) => ({ index, value })); const value = kpi.value === null ? '—' : kpi.key === 'average_change' ? `${kpi.value > 0 ? '+' : ''}${kpi.value}` : String(kpi.value); return <div key={kpi.key} className="rounded-xl border border-border bg-card p-4 shadow-[0_1px_2px_rgba(16,24,40,0.05)]"><div className="flex items-start justify-between gap-2"><p className="text-sm font-medium text-muted-foreground">{item.label}</p><span className={cn('grid size-9 shrink-0 place-items-center rounded-full', background[item.tone], text[item.tone])}><Icon className="size-[18px]" /></span></div><p className={cn('mt-1 text-3xl font-bold tracking-tight', text[item.tone])}>{value}</p><p className="mt-1 text-xs text-muted-foreground">{kpi.delta}</p><div className="mt-2 h-10"><ResponsiveContainer width="100%" height="100%">{item.chart === 'bar' ? <BarChart data={data}><Bar dataKey="value" fill={stroke[item.tone]} radius={[2, 2, 0, 0]} isAnimationActive={false} /></BarChart> : <AreaChart data={data}><Area type="monotone" dataKey="value" stroke={stroke[item.tone]} strokeWidth={1.6} fill="transparent" dot={false} isAnimationActive={false} /></AreaChart>}</ResponsiveContainer></div></div>; })}</div>; }
