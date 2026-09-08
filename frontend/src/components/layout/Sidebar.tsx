import { FolderKanban, LayoutDashboard, Target, TriangleAlert, X } from 'lucide-react';
import { NavLink } from 'react-router-dom';

const nav = [
  ['/dashboard', 'Dashboard', LayoutDashboard],
  ['/projects', 'Projects', FolderKanban],
  ['/early-warnings', 'Early Warnings', TriangleAlert],
  ['/prediction-accuracy', 'Prediction Accuracy', Target],
] as const;

export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return <>
    <button type="button" aria-label="Close navigation" tabIndex={open ? 0 : -1} onClick={onClose} className={`fixed inset-0 z-40 bg-slate-950/40 transition-opacity ${open ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`} />
    <aside aria-label="Application navigation" aria-hidden={!open} className={`fixed inset-y-0 left-0 z-50 flex w-[min(280px,86vw)] flex-col bg-[#102a43] text-slate-300 shadow-2xl transition-transform duration-200 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
      <div className="flex items-start justify-between border-b border-white/10 px-6 py-6">
        <div><div className="text-xl font-bold text-white">InfraSight <span className="text-blue-400">AI</span></div><p className="mt-1 text-[10px] tracking-widest text-slate-400">INFRASTRUCTURE INTELLIGENCE</p></div>
        <button type="button" aria-label="Close sidebar" onClick={onClose} className="grid size-9 place-items-center rounded-md text-slate-300 hover:bg-white/10 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400"><X size={20} /></button>
      </div>
      <nav aria-label="Primary navigation" className="flex-1 px-3 py-5">{nav.map(([path, label, Icon]) => <NavLink key={path} to={path} onClick={onClose} className={({ isActive }) => `mb-1 flex items-center gap-3 rounded-md px-3 py-2.5 text-sm ${isActive ? 'bg-blue-600 text-white shadow-sm' : 'hover:bg-white/10'}`}><Icon size={17} />{label}</NavLink>)}</nav>
    </aside>
  </>;
}
