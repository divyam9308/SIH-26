import { Link } from 'react-router-dom';
import type { Project } from '../../types/dashboard';
import { Card } from '../ui/Card';

export function InterventionQueue({ projects }: { projects: Project[] }) {
  return <Card className="overflow-hidden p-0">
    <div className="border-b border-slate-100 px-4 py-3"><h2 className="text-xs font-bold tracking-wide text-slate-700">TOP PRIORITY: INTERVENTION QUEUE</h2></div>
    <div className="overflow-x-auto"><table className="w-full min-w-160 text-left text-xs"><thead className="bg-slate-50 text-[10px] uppercase tracking-wide text-slate-500"><tr>{['#', 'Project Name', 'Sector', 'Risk Score', 'Cost overrun', 'Delay', 'Progress'].map((header) => <th key={header} className="whitespace-nowrap px-3 py-2.5 font-bold">{header}</th>)}</tr></thead><tbody>{projects.map((project) => <tr key={project.code ?? project.id} className="border-t border-slate-100"><td className="px-3 py-3 text-slate-500">{project.id}</td><td className="px-3 py-3 font-semibold text-slate-700">{project.code ? <Link to={`/projects/${project.code}`}>{project.name}</Link> : project.name}</td><td className="px-3 py-3 text-slate-600">{project.sector}</td><td className="px-3 py-3"><span className="font-bold text-red-600">{project.riskScore}</span></td><td className="px-3 py-3 text-slate-600">{project.costRisk.toFixed(1)}%</td><td className="px-3 py-3 text-slate-600">{project.scheduleRiskDays.toFixed(0)} days</td><td className="px-3 py-3 text-slate-600">{project.progress === null ? 'Not reported' : `${project.progress}%`}</td></tr>)}</tbody></table></div>
    <Link to="/projects" className="block border-t border-slate-100 px-4 py-3 text-xs font-semibold text-blue-600">View all projects →</Link>
  </Card>;
}
