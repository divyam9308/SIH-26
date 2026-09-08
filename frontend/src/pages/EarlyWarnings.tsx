import { useEffect, useState } from 'react';
import { Bell, Calendar, ChevronDown, Download, Info, Search, SlidersHorizontal } from 'lucide-react';
import { KpiCards } from '../components/early-warnings/KpiCards';
import { WarningQueue } from '../components/early-warnings/WarningQueue';
import { EmergingDriversCard, WarningTrendCard } from '../components/early-warnings/InsightCards';
import { EscalationTimeline, RecentAlerts } from '../components/early-warnings/BottomPanels';
import { EarlyWarningsSwitch } from '../components/ui/EarlyWarningsSwitch';
import { cn } from '../lib/earlyWarningsUtils';
import { getEarlyWarnings } from '../services/earlyWarningsService';
import type { EarlyWarningsResponse, WarningStatus } from '../types/earlyWarnings';
import './EarlyWarnings.css';

const PAGE_SIZE = 15;
const severities: Array<WarningStatus | 'All'> = ['All', 'New Escalation', 'Worsening', 'Persistent', 'Improving'];
const severityLabel: Record<WarningStatus | 'All', string> = { All: 'All', 'New Escalation': 'New', Worsening: 'Worsening', Persistent: 'Persistent', Improving: 'Improving' };
const snapshotLabel = (value: string) => new Intl.DateTimeFormat('en-IN', { month: 'short', year: 'numeric' }).format(new Date(`${value}T00:00:00`));

export function EarlyWarnings() {
  const [severity, setSeverity] = useState<WarningStatus | 'All'>('All');
  const [sector, setSector] = useState('All Sectors');
  const [ministry, setMinistry] = useState('All Ministries');
  const [search, setSearch] = useState('');
  const [thresholdOnly, setThresholdOnly] = useState(true);
  const [asOf, setAsOf] = useState<string>();
  const [page, setPage] = useState(1);
  const [data, setData] = useState<EarlyWarningsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    document.title = 'Early Warnings — InfraSight AI / PAIMANA AI';
    document.querySelector('meta[name="description"]')?.setAttribute('content', 'Detect newly escalating infrastructure projects using monthly risk changes, forecast shifts, and model feature drivers.');
  }, []);
  useEffect(() => { setPage(1); }, [asOf, search, severity, sector, ministry, thresholdOnly]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null);
    getEarlyWarnings({ asOf, search, severity, sector, ministry, thresholdOnly }, controller.signal)
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return;
        setData(null); setError(reason instanceof Error ? reason.message : 'Early warning data could not be loaded.');
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [asOf, search, severity, sector, ministry, thresholdOnly]);

  const snapshot = data?.snapshot_date ?? asOf;
  const warningRows = data?.warnings ?? [];
  const pageCount = Math.max(1, Math.ceil(warningRows.length / PAGE_SIZE));
  const pagedRows = warningRows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return <div className="early-warnings-page min-h-screen bg-surface font-sans text-foreground"><main className="min-w-0 px-5 py-5 lg:px-6">
    <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between"><div><h1 className="text-3xl font-bold tracking-tight">Early Warnings</h1><p className="mt-1 max-w-lg text-sm text-muted-foreground">Detect newly escalating infrastructure projects using monthly risk changes, forecast shifts, and model feature drivers.</p></div><div className="flex flex-wrap items-center gap-3"><div className="relative inline-flex h-11 items-center gap-2 rounded-lg border border-border bg-card px-3 text-sm font-medium"><Calendar className="size-4 text-muted-foreground" /><select aria-label="Snapshot month" value={snapshot ?? ''} onChange={(event) => setAsOf(event.target.value)} disabled={!data} className="appearance-none bg-transparent pr-8 outline-none disabled:cursor-wait">{!snapshot && <option>Loading…</option>}{data?.available_snapshots.map((value) => <option key={value} value={value}>{snapshotLabel(value)}</option>)}</select><ChevronDown className="pointer-events-none absolute right-3 size-4 text-muted-foreground" /></div><div className="relative"><input value={search} onChange={(event) => setSearch(event.target.value)} type="search" placeholder="Search projects, sectors, drivers..." className="h-11 w-full min-w-56 rounded-lg border border-border bg-card pl-3 pr-9 text-sm outline-none placeholder:text-muted-foreground focus:ring-2 focus:ring-ring/30 sm:w-[300px]" /><Search className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /></div><button type="button" className="inline-flex h-11 items-center gap-2 rounded-lg border border-border bg-card px-3 text-sm font-medium"><SlidersHorizontal className="size-4 text-muted-foreground" />Filters<ChevronDown className="ml-6 size-4 text-muted-foreground" /></button><button aria-label="Notifications" className="relative grid size-10 place-items-center rounded-full text-muted-foreground hover:bg-card"><Bell className="size-5" /><span className="absolute right-1.5 top-1.5 size-2 rounded-full bg-primary" /></button><div className="flex items-center gap-2 pr-1"><span className="grid size-9 place-items-center rounded-full bg-info-soft text-sm font-semibold text-info">AN</span><div className="leading-tight"><p className="text-sm font-semibold">Arjun N.</p><p className="text-xs text-muted-foreground">Risk Analyst</p></div><ChevronDown className="size-4 text-muted-foreground" /></div></div></div>
    <div className="mt-5"><KpiCards kpis={data?.summary.kpis} /></div>{error && <p role="alert" className="mt-3 text-sm text-critical">{error}</p>}{loading && !data && <p className="mt-3 text-sm text-muted-foreground">Loading verified PAIMANA warning data…</p>}
    <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-3"><span className="text-sm font-semibold">Severity</span><div className="flex flex-wrap gap-2">{severities.map((item) => <button key={item} onClick={() => setSeverity(item)} className={cn('rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors', severity === item ? 'border-primary bg-primary text-primary-foreground' : 'border-border bg-card text-foreground hover:bg-accent')}>{severityLabel[item]}</button>)}</div><span className="ml-1 text-sm font-semibold">Sector</span><select aria-label="Sector" value={sector} onChange={(event) => setSector(event.target.value)} className="h-9 min-w-[150px] rounded-lg border border-border bg-card px-3 text-sm text-muted-foreground"><option>All Sectors</option>{data?.available_filters.sectors.map((value) => <option key={value}>{value}</option>)}</select><span className="ml-1 text-sm font-semibold">Ministry</span><select aria-label="Ministry" value={ministry} onChange={(event) => setMinistry(event.target.value)} className="h-9 min-w-[150px] rounded-lg border border-border bg-card px-3 text-sm text-muted-foreground"><option>All Ministries</option>{data?.available_filters.ministries.map((value) => <option key={value}>{value}</option>)}</select><div className="flex items-center gap-2"><EarlyWarningsSwitch checked={thresholdOnly} onCheckedChange={setThresholdOnly} /><span className="text-sm text-muted-foreground">Only threshold-crossing warnings</span><Info className="size-4 text-muted-foreground" /></div><button type="button" className="ml-auto inline-flex h-10 items-center gap-2 rounded-lg border border-border bg-card px-3 text-sm font-medium"><Download className="size-4 text-muted-foreground" />Export<ChevronDown className="ml-4 size-4 text-muted-foreground" /></button></div>
    <div className="mt-4 grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]"><WarningQueue rows={pagedRows} page={page} pageCount={pageCount} total={warningRows.length} pageSize={PAGE_SIZE} onPageChange={setPage} /><div className="flex flex-col gap-4"><EmergingDriversCard drivers={data?.drivers} /><WarningTrendCard trend={data?.trend} /></div></div>
    <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2"><EscalationTimeline timeline={data?.timeline} /><RecentAlerts alerts={data?.recent_alerts} /></div>
  </main></div>;
}
