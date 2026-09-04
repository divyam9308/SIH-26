import { Header } from '../components/layout/Header';
import { FilterBar } from '../components/dashboard/FilterBar';
import { KpiCard } from '../components/dashboard/KpiCard';
import { InterventionQueue } from '../components/dashboard/InterventionQueue';
import { PortfolioRiskDistribution } from '../components/dashboard/PortfolioRiskDistribution';
import { ExpenditureProgressChart } from '../components/dashboard/ExpenditureProgressChart';
import { WarningDriversChart } from '../components/dashboard/WarningDriversChart';
import { Card } from '../components/ui/Card';
import { useEffect, useState } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';
import { useFilters } from '../hooks/useFilters';
import { SAVED_WINDOW_STORAGE_KEY, SAVED_WINDOWS } from '../components/dashboard/FilterBar';

export function Dashboard() {
  const [window, setWindow] = useState(() => {
    const stored = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY);
    return stored && SAVED_WINDOWS.some((item) => item.key === stored) ? stored : '2001_2017';
  });
  useEffect(() => { globalThis.localStorage?.setItem(SAVED_WINDOW_STORAGE_KEY, window); }, [window]);
  const { filters, setFilter, resetFilters } = useFilters();
  const { data, loading, error, refresh } = useDashboardData(window);
  const sectors = data ? [...new Set(data.projects.map((project) => project.sector))].sort() : [];
  const projects = data?.projects.filter((project) =>
    (filters.sector === 'All Sectors' || project.sector === filters.sector) &&
    (filters.riskLevel === 'All Levels' || project.riskLevel === filters.riskLevel.toUpperCase())
  ) ?? [];
  return <>
    <Header onRefresh={refresh} datasetSnapshot={data?.datasetSnapshot ?? null} available={!error && Boolean(data)} />
    <FilterBar filters={filters} setFilter={setFilter} reset={resetFilters} sectors={sectors} window={window} onWindowChange={setWindow} />
    <div className="p-6">
      {loading && !data && <Card>Loading real PAIMANA portfolio predictions…</Card>}
      {error && <Card className="border-red-200 text-sm text-red-700"><b>Backend data unavailable.</b><p className="mt-1">{error}</p><button className="mt-3 text-blue-700" onClick={refresh}>Try again</button></Card>}
      {data && <>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">{data.kpis.map((item) => <KpiCard key={item.title} item={item} />)}</div>
        <div className="mt-5 grid grid-cols-1 gap-5 xl:grid-cols-3"><div className="xl:col-span-2"><InterventionQueue projects={projects} /></div><PortfolioRiskDistribution data={data.riskDistribution} total={data.totalProjects ?? 0} /></div>
        <div className="mt-5 grid grid-cols-1 gap-5 xl:grid-cols-3">
          <Card><h2 className="text-xs font-bold tracking-wide text-slate-700">RISK TREND</h2><div className="flex h-48 items-center justify-center px-6 text-center text-xs text-slate-500">{data.riskTrendStatus ?? 'Unavailable'}</div></Card>
          <ExpenditureProgressChart data={data.expenditureProgress} />
          <WarningDriversChart data={data.warningDrivers} />
        </div>
        <p className="mt-6 text-center text-[11px] text-slate-500">Model {data.modelVersion ?? 'Unavailable'} · dataset snapshot {data.datasetSnapshot ?? 'Unavailable'} · {data.modelScope ?? 'Model scope unavailable'}</p>
      </>}
    </div>
  </>;
}
