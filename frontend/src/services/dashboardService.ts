import { apiGet } from './api';
import type { DashboardData } from '../types/dashboard';
import type { PortfolioRiskItem, PortfolioSummaryResponse } from '../types/api';

const inr = (value: number) => `₹${new Intl.NumberFormat('en-IN', { maximumFractionDigits: 1 }).format(value)} Cr`;

export async function getDashboardData(window = '2001_2017', signal?: AbortSignal): Promise<DashboardData> {
  const [summary, risk] = await Promise.all([
    apiGet<PortfolioSummaryResponse>(`/api/portfolio/summary?window=${encodeURIComponent(window)}`, signal),
    apiGet<{ items: PortfolioRiskItem[] }>(`/api/portfolio/risk?limit=20&window=${encodeURIComponent(window)}`, signal),
  ]);
  const highCritical = summary.risk_distribution.high + summary.risk_distribution.critical;
  const colors: Record<string, string> = { Critical: '#dc2626', High: '#ea580c', Medium: '#ca8a04', Low: '#16a34a' };
  return {
    kpis: [
      { title: 'TOTAL PROJECTS', value: summary.projects.toLocaleString('en-IN'), change: `${summary.sectors} reported sectors`, tone: 'blue' },
      { title: 'HIGH / CRITICAL PROJECTS', value: highCritical.toLocaleString('en-IN'), change: 'Production risk classification', tone: 'red' },
      { title: 'PREDICTED COST EXPOSURE', value: inr(summary.predicted_cost_exposure_cr), change: 'Positive predicted overruns only', tone: 'orange' },
      { title: 'CURRENT EXPENDITURE', value: inr(summary.expenditure_cr), change: `Snapshot ${summary.dataset_snapshot ?? 'Unavailable'}`, tone: 'blue' },
    ],
    projects: risk.items.map((item, index) => ({
      id: String(index + 1), code: item.project_code, name: item.project_name, sector: item.sector, riskLevel: item.risk_level,
      riskScore: item.risk_score, costRisk: item.predicted_cost_overrun_percentage,
      scheduleRisk: item.predicted_delay_months, progress: item.physical_progress_pct,
      warning: item.confidence_calibration_status.replaceAll('_', ' '),
    })),
    riskDistribution: (['Critical', 'High', 'Medium', 'Low'] as const).map((name) => ({ name, value: summary.risk_distribution[name.toLowerCase() as keyof typeof summary.risk_distribution], color: colors[name] })),
    totalProjects: summary.projects,
    expenditureProgress: summary.expenditure_progress.map((item) => ({ x: item.physical_progress_pct, y: item.financial_progress_pct, group: item.group })),
    warningDrivers: summary.warning_drivers.map((item) => ({ name: item.name, value: item.count })),
    riskTrendStatus: summary.risk_trend_status,
    modelVersion: summary.model_version, datasetSnapshot: summary.dataset_snapshot,
    modelScope: summary.model_scope, inferenceTimestamp: summary.inference_timestamp,
  };
}
