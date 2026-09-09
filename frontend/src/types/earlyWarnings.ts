export type WarningStatus = 'New Escalation' | 'Worsening' | 'Persistent' | 'Improving';

export interface RiskReading { level: 'Low' | 'Medium' | 'High' | 'Critical'; score: number; }
export interface EarlyWarningRow {
  project_id: string; project_name: string; sector: string; ministry: string | null; snapshot_date: string;
  previous_risk: RiskReading | null; current_risk: RiskReading; risk_change: number | null;
  predicted_cost_overrun_percentage: number | null; predicted_delay_months: number | null; predicted_delay_days: number | null;
  predicted_cost_overrun_amount_cr: number | null; warning_status: WarningStatus; threshold_crossing: boolean;
  first_appeared: string | null; priority: RiskReading['level']; trigger: string;
}
export interface EarlyWarningKpi { key: string; value: number | null; delta: string; data: number[]; }
export interface WarningTrendPoint { month: string; snapshot_date: string; newEsc: number; worsening: number; persistent: number; improving: number; average_risk_change: number; }
export interface EarlyWarningDriver { name: string; value: number; }
export interface EarlyWarningTimeline { project_id: string; project_name: string; points: { month: string; score: number }[]; risk_change: number | null; current_risk: RiskReading; }
export interface EarlyWarningAlert { project_name: string; date: string; text: string; }
export interface EarlyWarningsResponse {
  snapshot_date: string; available_snapshots: string[]; summary: { kpis: EarlyWarningKpi[]; total_projects: number; matching_projects: number };
  warnings: EarlyWarningRow[]; trend: WarningTrendPoint[]; drivers: EarlyWarningDriver[];
  drivers_metadata: { available: boolean; method?: string; scope?: string; source?: string; reason?: string };
  timeline: EarlyWarningTimeline | null; recent_alerts: EarlyWarningAlert[];
  available_filters: { sectors: string[]; ministries: string[] };
  source: { window: string; model_version: string; run_id: string; ledger: string; trajectories: string };
}
