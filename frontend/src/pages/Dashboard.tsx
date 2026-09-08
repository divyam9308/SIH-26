import { useEffect, useState } from 'react';
import { Header } from '../components/layout/Header';
import { FilterBar } from '../components/dashboard/FilterBar';
import { KpiCard } from '../components/dashboard/KpiCard';
import { InterventionQueue } from '../components/dashboard/InterventionQueue';
import { PortfolioRiskDistribution } from '../components/dashboard/PortfolioRiskDistribution';
import { Card } from '../components/ui/Card';
import { useDashboardData } from '../hooks/useDashboardData';
import { useFilters } from '../hooks/useFilters';
import { getDataRangeLabel, SAVED_WINDOW_STORAGE_KEY, SAVED_WINDOWS } from '../lib/trainingWindows';

export function Dashboard() {
  const [window, setWindow] = useState(() => {
    const stored = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY);
    return stored && SAVED_WINDOWS.some((item) => item.key === stored) ? stored : '2001_2017';
  });
  useEffect(() => { globalThis.localStorage?.setItem(SAVED_WINDOW_STORAGE_KEY, window); }, [window]);
  const { filters, setFilter } = useFilters();
  const { data, loading, error, refresh } = useDashboardData(window);
  const sectors = data ? [...new Set(data.projects.map((project) => project.sector))].sort() : [];
  const projects = data?.projects.filter((project) =>
    (filters.sector === 'All Sectors' || project.sector === filters.sector) &&
    (filters.riskLevel === 'All Levels' || project.riskLevel === filters.riskLevel.toUpperCase())
  ) ?? [];

  return <>
    <Header onRefresh={refresh} dataRange={getDataRangeLabel(window)} available={!error && Boolean(data)} />
    <FilterBar filters={filters} setFilter={setFilter} sectors={sectors} window={window} onWindowChange={setWindow} />
    <div className="p-4 sm:p-6">
      {loading && !data && <Card>Loading real PAIMANA portfolio predictions…</Card>}
      {error && <Card className="border-red-200 text-sm text-red-700"><b>Backend data unavailable.</b><p className="mt-1">{error}</p><button className="mt-3 text-blue-700" onClick={refresh}>Try again</button></Card>}
      {data && <>
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(340px,0.8fr)]">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">{data.kpis.map((item) => <KpiCard key={item.title} item={item} />)}</div>
          <PortfolioRiskDistribution data={data.riskDistribution} total={data.totalProjects ?? 0} />
        </div>
        <div className="mt-5"><InterventionQueue projects={projects} /></div>
        <p className="mt-6 text-center text-[11px] text-slate-500">Model {data.modelVersion ?? 'Unavailable'} · Data Range {getDataRangeLabel(window)} · {data.modelScope ?? 'Model scope unavailable'}</p>
      </>}
    </div>
  </>;
}
