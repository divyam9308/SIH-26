import { useEffect, useState } from 'react';
import { ArrowUpDown, ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { getProjects, type ProjectSort } from '../services/projectService';
import type { ProjectListResponse, RiskLevel } from '../types/api';
import '../styles/projects.css';
import { SAVED_WINDOW_STORAGE_KEY, SAVED_WINDOWS } from '../components/dashboard/FilterBar';

export type RiskCategory = 'Critical' | 'High' | 'Medium' | 'Low';
const categories: RiskCategory[] = ['Critical', 'High', 'Medium', 'Low'];
const descriptions: Record<RiskCategory, string> = { Critical: 'Immediate intervention', High: 'Escalating exposure', Medium: 'Active monitoring', Low: 'Broadly on track' };
const PAGE_SIZE = 10;
export const riskClass = (risk: RiskCategory) => risk.toLowerCase();
export const displayRisk = (risk: RiskLevel): RiskCategory => `${risk[0]}${risk.slice(1).toLowerCase()}` as RiskCategory;
export const inr = (value: number | null) => value === null ? 'Not reported' : `₹${new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 }).format(value)} Cr`;
const percentage = (value: number | null | undefined) => value == null ? 'Not reported' : `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
const days = (value: number | null | undefined) => value == null ? 'Not reported' : `${value.toFixed(0)} days`;
const costErrorLabel = (value: number | null | undefined) => value == null ? 'Error unavailable' : `${value.toFixed(1)} pp error`;
const delayErrorLabel = (value: number | null | undefined) => value == null ? 'Error unavailable' : `${value.toFixed(0)} days error`;

export function RiskChip({ level, className = '' }: { level: RiskCategory; className?: string }) {
  return <span className={`project-risk-chip ${riskClass(level)} ${className}`}><span />{level}</span>;
}

export function ProjectPanel({ title, subtitle, action, children, className = '' }: { title?: string; subtitle?: string; action?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return <section className={`project-panel ${className}`}>{title && <header className="project-panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</header>}{children}</section>;
}

export function Projects() {
  const navigate = useNavigate();
  const [active, setActive] = useState<RiskCategory | null>(null);
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState<{ key: ProjectSort; direction: 'asc' | 'desc' }>({ key: 'score', direction: 'desc' });
  const [selectedWindow, setSelectedWindow] = useState(() => {
    const stored = globalThis.localStorage?.getItem(SAVED_WINDOW_STORAGE_KEY);
    return stored && SAVED_WINDOWS.some((item) => item.key === stored) ? stored : '2001_2017';
  });
  useEffect(() => { globalThis.localStorage?.setItem(SAVED_WINDOW_STORAGE_KEY, selectedWindow); }, [selectedWindow]);
  const [data, setData] = useState<ProjectListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { const timer = window.setTimeout(() => { setDebouncedQuery(query.trim()); setPage(1); }, 250); return () => window.clearTimeout(timer); }, [query]);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError(null);
    getProjects({ page, pageSize: PAGE_SIZE, search: debouncedQuery || undefined, riskLevel: active ?? undefined, sort: sort.key, direction: sort.direction, window: selectedWindow }, controller.signal)
      .then(setData).catch((reason: unknown) => { if (reason instanceof DOMException && reason.name === 'AbortError') return; setError(reason instanceof Error ? reason.message : 'Projects are unavailable.'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [active, debouncedQuery, page, sort, selectedWindow]);
  const toggleSort = (key: ProjectSort) => { setPage(1); setSort((previous) => previous.key === key ? { key, direction: previous.direction === 'asc' ? 'desc' : 'asc' } : { key, direction: ['name', 'code', 'sector'].includes(key) ? 'asc' : 'desc' }); };
  const header = (key: ProjectSort, label: string, align = '') => <th className={align}><button onClick={() => toggleSort(key)} className={sort.key === key ? 'is-sorted' : ''}>{label}<ArrowUpDown size={13} /></button></th>;
  const stats = categories.map((category) => { const key = category.toLowerCase() as keyof ProjectListResponse['risk_distribution']; return { category, count: data?.risk_distribution[key] ?? 0, exposure: data?.cost_exposure_by_risk_cr[key] ?? 0 }; });
  const rows = data?.items ?? [];
  const pageCount = data?.pages ?? 1;

  return <div className="projects-page">
    <div className="projects-heading">
      <div><p className="projects-eyebrow">Projects</p><h1>Implementation Risk Register</h1><p className="projects-summary">{data?.total ?? '—'} matching projects · dataset {data?.dataset_snapshot ?? 'Unavailable'} · model {data?.model_version ?? 'Unavailable'}</p></div><label className="project-window-selector">Saved training range<select value={selectedWindow} onChange={(event) => { setSelectedWindow(event.target.value); setPage(1); }}>{SAVED_WINDOWS.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label>
      <div className="portfolio-card"><p>Portfolio predicted cost exposure</p><strong>{inr(data?.predicted_cost_exposure_cr ?? null)}</strong></div>
    </div>
    <div className="risk-card-grid">{stats.map((stat) => { const selected = active === stat.category; return <button key={stat.category} aria-pressed={selected} className={`risk-card ${riskClass(stat.category)} ${selected ? 'is-active' : ''}`} onClick={() => { setActive(selected ? null : stat.category); setPage(1); }}><div><b>{stat.category}</b><i /></div><strong>{String(stat.count).padStart(2, '0')}</strong><p>{descriptions[stat.category]}</p><div className="risk-progress"><span style={{ width: `${data ? (stat.count / Math.max(1, Object.values(data.risk_distribution).reduce((a, b) => a + b, 0))) * 100 : 0}%` }} /></div><small>{inr(stat.exposure)} predicted overrun</small></button>; })}</div>
    <ProjectPanel title="Projects" subtitle={active ? `Filtered to ${active} risk · ${data?.total ?? 0} projects` : `All risk levels · ${data?.total ?? 0} projects`} className="projects-register" action={<div className="register-actions">{active && <button className="clear-filter" onClick={() => { setActive(null); setPage(1); }}>Clear filter</button>}<label className="project-search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name, code, sector…" aria-label="Search projects" /></label></div>}>
      {error && <div className="project-state error"><b>Backend unavailable</b><p>{error}</p></div>}
      {loading && !data && <div className="project-state">Loading real project predictions…</div>}
      {!error && <div className="project-table-wrap"><table className="project-table"><thead><tr>{header('name', 'Project Name')}{header('code', 'Project Code')}{header('sector', 'Sector')}{header('score', 'Severity score', 'numeric')}{header('cost', 'Predicted Cost Overrun', 'numeric')}<th className="numeric">Actual Cost Overrun</th>{header('time', 'Predicted Time Overrun', 'numeric')}<th className="numeric">Actual Time Overrun</th><th>Risk Category</th></tr></thead><tbody>{rows.map((project) => { const category = displayRisk(project.risk_level); return <tr key={project.project_code} tabIndex={0} role="link" aria-label={`Open ${project.project_name} project intelligence`} onClick={() => navigate(`/projects/${project.project_code}`)} onKeyDown={(event) => { if (event.key === 'Enter') navigate(`/projects/${project.project_code}`); }}><td><div className="project-name"><span className={`risk-marker ${riskClass(category)}`} />{project.project_name}</div></td><td className="project-code">{project.project_code}</td><td className="project-sector">{project.sector}</td><td className="numeric"><b>{project.risk_score.toFixed(1)}</b><small>/100</small></td><td className="numeric project-overrun"><b className={riskClass(category)}>{percentage(project.predicted_cost_overrun_percentage)}</b><small>{inr(project.predicted_cost_overrun_amount_cr)}</small></td><td className="numeric project-overrun"><b>{percentage(project.actual_cost_overrun_percentage)}</b><small>{costErrorLabel(project.cost_error_percentage)}</small></td><td className="numeric project-overrun"><b className={riskClass(category)}>+{project.predicted_delay_months.toFixed(1)} mo</b><small>{project.physical_progress_pct === null ? 'Physical progress not reported' : `${project.physical_progress_pct}% physical`}</small></td><td className="numeric project-overrun"><b>{days(project.actual_delay_days)}</b><small>{delayErrorLabel(project.delay_error_days)}</small></td><td><div className="project-category"><RiskChip level={category} /><ChevronRight size={16} /></div></td></tr>; })}{!loading && rows.length === 0 && <tr><td className="project-empty" colSpan={9}>No real projects match this search and filter.</td></tr>}</tbody></table></div>}
      {!error && <div className="project-pagination"><p>Showing <b>{data?.total === 0 ? 0 : ((data?.page ?? 1) - 1) * PAGE_SIZE + 1}–{Math.min((data?.page ?? 1) * PAGE_SIZE, data?.total ?? 0)}</b> of <b>{data?.total ?? 0}</b> projects</p><div><button onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={(data?.page ?? 1) === 1} aria-label="Previous page"><ChevronLeft size={16} /></button>{Array.from({ length: pageCount }, (_, index) => index + 1).map((value) => <button key={value} onClick={() => setPage(value)} aria-current={value === (data?.page ?? 1) ? 'page' : undefined} className={value === (data?.page ?? 1) ? 'is-current' : ''}>{value}</button>)}<button onClick={() => setPage((value) => Math.min(pageCount, value + 1))} disabled={(data?.page ?? 1) === pageCount} aria-label="Next page"><ChevronRight size={16} /></button></div></div>}
    </ProjectPanel>
  </div>;
}
