import { useEffect, useState } from 'react';
import type { DashboardFilters } from '../../types/filters';
import { getSavedWindow, getTrainingWindowLabel, SAVED_WINDOWS } from '../../lib/trainingWindows';
import { Select } from '../ui/Select';

export { SAVED_WINDOW_STORAGE_KEY, SAVED_WINDOWS } from '../../lib/trainingWindows';

interface FilterBarProps {
  filters: DashboardFilters;
  setFilter: <K extends keyof DashboardFilters>(key: K, value: DashboardFilters[K]) => void;
  sectors: string[];
  window: string;
  onWindowChange: (window: string) => void;
}

export function FilterBar({ filters, setFilter, sectors, window, onWindowChange }: FilterBarProps) {
  const active = getSavedWindow(window);
  const [from, setFrom] = useState(String(active.from));
  const [to, setTo] = useState(String(active.to));
  const [message, setMessage] = useState('');

  useEffect(() => {
    setFrom(String(active.from));
    setTo(String(active.to));
  }, [active.from, active.to]);

  const apply = () => {
    const key = `${Number(from)}_${Number(to)}`;
    if (!SAVED_WINDOWS.some((item) => item.key === key)) {
      setMessage('Choose one of the three saved ranges.');
      return;
    }
    setMessage('');
    onWindowChange(key);
  };

  return <div className="flex flex-wrap items-end gap-3 border-b border-slate-200 bg-white px-6 py-3">
    <Select label="Sector" value={filters.sector} onChange={(value) => setFilter('sector', value)} options={['All Sectors', ...sectors]} />
    <Select label="Risk Level" value={filters.riskLevel} onChange={(value) => setFilter('riskLevel', value)} options={['All Levels', 'Critical', 'High', 'Medium', 'Low']} />
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Training from<input className="mt-1 block h-10 w-24 rounded-md border border-slate-300 px-3 text-xs text-slate-700 outline-none focus:border-blue-500" type="number" value={from} onChange={(event) => setFrom(event.target.value)} /></label>
    <label className="text-[10px] font-bold uppercase tracking-wide text-slate-500">Training to<input className="mt-1 block h-10 w-24 rounded-md border border-slate-300 px-3 text-xs text-slate-700 outline-none focus:border-blue-500" type="number" value={to} onChange={(event) => setTo(event.target.value)} /></label>
    <button type="button" onClick={apply} className="h-10 rounded-md border border-slate-800 bg-slate-800 px-4 text-xs font-semibold text-white hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600">Load saved range</button>
    {message && <span className="basis-full text-xs text-amber-700" role="alert">{message}</span>}
    <span className="basis-full text-[11px] text-slate-500">{getTrainingWindowLabel(window)} · frozen production evaluation, no retraining</span>
  </div>;
}
