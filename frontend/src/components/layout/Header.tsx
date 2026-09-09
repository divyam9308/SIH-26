import { RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';

export function Header({ onRefresh, dataRange, available }: { onRefresh: () => void; dataRange: string; available: boolean }) {
  return <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-6 py-3 pl-17 sm:pl-18">
    <div className="text-xs text-slate-500"><span className="font-semibold text-slate-700">Data Range:</span> {dataRange} <span className="mx-3 text-slate-300">|</span><span className="font-semibold text-slate-700">Data Source:</span> official PAIMANA public-project subset</div>
    <div className="flex items-center gap-3"><span className={`flex items-center gap-1.5 text-xs font-semibold ${available ? 'text-emerald-600' : 'text-red-600'}`}><span className={`h-2 w-2 rounded-full ${available ? 'bg-emerald-500' : 'bg-red-500'}`} />{available ? 'Live' : 'Unavailable'}</span><Button onClick={onRefresh} className="py-1.5"><RefreshCw size={14} />Refresh</Button></div>
  </header>;
}
