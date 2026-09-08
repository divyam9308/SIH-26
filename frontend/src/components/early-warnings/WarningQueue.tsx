import { ArrowDown, ArrowUp, Building2, Info } from 'lucide-react';
import { Pill, RiskText, type Tone } from '../dashboard/EarlyWarningsPill';
import { cn } from '../../lib/earlyWarningsUtils';
import type { EarlyWarningRow, WarningStatus } from '../../types/earlyWarnings';

const status: Record<WarningStatus, Tone> = { 'New Escalation': 'critical', Worsening: 'high', Persistent: 'medium', Improving: 'success' };
const priority: Record<EarlyWarningRow['priority'], Tone> = { Critical: 'critical', High: 'high', Medium: 'medium', Low: 'low' };
const headers = ['Project', 'Sector', 'Previous Risk', 'Current Risk', 'Risk Change', 'Pred. Cost Overrun', 'Pred. Delay', 'Trigger / Driver', 'Warning Status', 'First Appeared', 'Priority'];
const percentage = (value: number | null) => value === null ? '—' : `${value.toFixed(1)}%`;
const delay = (value: number | null) => value === null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(0)} days`;
const date = (value: string | null) => value ? new Intl.DateTimeFormat('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }).format(new Date(`${value}T00:00:00`)) : '—';

interface WarningQueueProps {
  rows?: EarlyWarningRow[];
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

export function WarningQueue({ rows = [], page, pageCount, total, pageSize, onPageChange }: WarningQueueProps) {
  const first = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);
  return <section className="overflow-hidden rounded-xl border border-border bg-card shadow-[0_1px_2px_rgba(16,24,40,0.05)]">
    <header className="flex items-center gap-2 px-5 py-4"><h2 className="text-lg font-semibold text-foreground">Active Warning Queue</h2><Info className="size-4 text-muted-foreground" /></header>
    <div className="overflow-x-auto"><table className="w-full border-collapse text-[13px]"><thead><tr className="bg-surface">{headers.map((header, index) => <th key={header} className={cn('whitespace-nowrap border-y border-border px-1 py-2.5 text-[11px] font-semibold text-muted-foreground', index === 0 ? 'text-left' : 'text-center')}>{header}</th>)}</tr></thead><tbody>{rows.map((row) => <tr key={`${row.project_id}-${row.snapshot_date}`} className="border-b border-border last:border-0 hover:bg-surface"><td className="w-[150px] px-2 py-2.5"><div className="flex items-start gap-2"><span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-md bg-surface text-muted-foreground"><Building2 className="size-4" /></span><span className="warning-project-name font-medium text-foreground" title={row.project_name}>{row.project_name}</span></div></td><td className="whitespace-nowrap px-1.5 py-2.5 text-center text-xs text-muted-foreground">{row.sector}</td><td className="px-1 py-2.5 text-center">{row.previous_risk ? <RiskText level={row.previous_risk.level} score={row.previous_risk.score} /> : '—'}</td><td className="px-1 py-2.5 text-center"><RiskText level={row.current_risk.level} score={row.current_risk.score} /></td><td className="px-1 py-2.5 text-center">{row.risk_change === null ? '—' : <span className={cn('inline-flex items-center gap-1 text-sm font-semibold', row.risk_change > 0 ? 'text-critical' : row.risk_change < 0 ? 'text-success' : 'text-foreground')}>{row.risk_change > 0 ? <ArrowUp className="size-3.5" /> : row.risk_change < 0 ? <ArrowDown className="size-3.5" /> : null}{Math.abs(row.risk_change)} pts</span>}</td><td className={cn('px-2 py-2.5 text-center font-medium', (row.predicted_cost_overrun_percentage ?? 0) >= 15 ? 'text-critical' : 'text-foreground')}>{percentage(row.predicted_cost_overrun_percentage)}</td><td className={cn('whitespace-nowrap px-2 py-2.5 text-center font-medium', (row.predicted_delay_days ?? 0) > 0 ? 'text-critical' : 'text-success')}>{delay(row.predicted_delay_days)}</td><td className="w-[100px] px-1.5 py-2.5 text-[11px] leading-snug text-muted-foreground"><span className="warning-trigger" title={row.trigger}>{row.trigger}</span></td><td className="px-1 py-2.5 text-center"><Pill tone={status[row.warning_status]}>{row.warning_status}</Pill></td><td className="whitespace-nowrap px-1.5 py-2.5 text-center text-xs text-muted-foreground">{date(row.first_appeared)}</td><td className="px-1 py-2.5 text-center"><Pill tone={priority[row.priority]}>{row.priority}</Pill></td></tr>)}{rows.length === 0 && <tr><td colSpan={headers.length} className="px-5 py-8 text-center text-sm text-muted-foreground">No real projects match the selected filters.</td></tr>}</tbody></table></div>
    <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-5 py-3 text-sm text-muted-foreground"><span>Showing {first}–{last} of {total} warnings</span><div className="flex items-center gap-2"><button type="button" onClick={() => onPageChange(page - 1)} disabled={page === 1} className="rounded-md border border-border px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40">&lt; Previous</button><span aria-live="polite">Page {page} of {pageCount}</span><button type="button" onClick={() => onPageChange(page + 1)} disabled={page === pageCount} className="rounded-md border border-border px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40">Next &gt;</button></div></footer>
  </section>;
}
