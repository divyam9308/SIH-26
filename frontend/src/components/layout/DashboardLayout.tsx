import { useEffect, useState, type ReactNode } from 'react';
import { Menu } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';

const pageTitle = (pathname: string) => {
  if (pathname.startsWith('/projects/')) return 'Project Details';
  if (pathname === '/projects') return 'Projects';
  if (pathname === '/early-warnings') return 'Early Warnings';
  if (pathname === '/prediction-accuracy') return 'Prediction Accuracy';
  if (pathname === '/model-comparison/training-window-performance') return 'Training Window Performance';
  return 'Dashboard';
};

export function DashboardLayout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => { setOpen(false); }, [location.pathname]);
  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false); };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [open]);

  return <div className="min-h-screen">
    <button type="button" aria-label="Open navigation" aria-expanded={open} onClick={() => setOpen(true)} className="fixed left-4 top-4 z-30 grid size-10 place-items-center rounded-lg border border-slate-200 bg-white text-slate-700 shadow-md hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"><Menu size={21} /></button>
    <div className="pointer-events-none flex h-14 items-center pl-17 sm:pl-18">
      <span className="text-lg font-semibold tracking-tight text-slate-800">{pageTitle(location.pathname)}</span>
    </div>
    <Sidebar open={open} onClose={() => setOpen(false)} />
    <main className="min-w-0 w-full">{children}</main>
  </div>;
}
