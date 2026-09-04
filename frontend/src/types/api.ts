export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface ProjectRecord {
  snapshot_date: string; sector: string; ministry: string | null; implementing_agency: string | null;
  project_code: string; project_name: string; original_cost_cr: number; revised_cost_cr: number | null;
  expenditure_cr: number | null; original_end_date: string; revised_end_date: string | null;
  physical_progress_pct: number | null; source_url: string; days_to_original_deadline: number;
  schedule_extension_days: number | null; cost_escalation_pct: number | null;
  expenditure_to_original_pct: number | null; financial_progress_pct: number | null;
  schedule_overrun_90d: number | null; cost_overrun_5pct: number | null;
  dq_expenditure_gt_revised: number; dq_revised_date_before_original: number;
  dq_missing_revised_cost: number; dq_missing_revised_date: number; dq_missing_progress: number;
}

export interface ShapFactor { feature: string; impact: number; direction: string }
export interface CapabilityStatus { available: boolean; reason: string | null; source: string | null }
export interface ForecastResponse {
  project_id: string; project_name: string; model_version: string; dataset_snapshot_date: string; inference_timestamp: string;
  current_status: { snapshot_month: string; physical_progress_percentage: number | null; current_estimated_cost: number | null; expenditure_cr: number | null; planned_completion_date: string; progress_delay_percentage_points: number | null };
  predicted_cost_overrun_percentage: number; predicted_cost_overrun_amount_cr: number; predicted_final_cost_cr: number;
  predicted_delay_days: number; predicted_cost_overrun: number; predicted_completion_date: string;
  current_progress: number | null; predicted_delay_months: number; risk_score: number;
  risk_probability_percentage: number; risk_level: RiskLevel; model_confidence_percentage: number | null;
  confidence_calibration_status: string; explanation: ShapFactor[]; shap_explanation: ShapFactor[];
  cost_factors: ShapFactor[]; delay_factors: ShapFactor[]; risk_factors: ShapFactor[];
  cost_explanation_status: CapabilityStatus; delay_explanation_status: CapabilityStatus; risk_explanation_status: CapabilityStatus;
  best_models: { cost: string; delay: string };
  expected_range: { cost_overrun_percentage: { p10: number; p50: number; p90: number }; delay_days: { p10: number; p50: number; p90: number } } | null;
  completion_probabilities: { year: number; probability_percentage: number }[]; features_used: string[]; model_scope: string;
}

export interface ProjectListItem {
  project_code: string; project_name: string; sector: string; ministry: string | null; implementing_agency: string | null;
  snapshot_date: string; original_cost_cr: number | null; revised_cost_cr: number | null; expenditure_cr: number | null;
  physical_progress_pct: number | null; predicted_cost_overrun_percentage: number; predicted_cost_overrun_amount_cr: number;
  predicted_final_cost_cr: number; predicted_delay_days: number; predicted_delay_months: number; predicted_completion_date: string | null;
  actual_cost_overrun_percentage?: number | null; actual_delay_days?: number | null; cost_error_percentage?: number | null; delay_error_days?: number | null;
  risk_score: number; risk_probability_percentage: number; risk_level: RiskLevel; model_version: string; model_scope: string;
  inference_timestamp: string; model_confidence_percentage: number | null; confidence_calibration_status: string;
}

export interface ProjectListResponse {
  items: ProjectListItem[]; total: number; page: number; page_size: number; pages: number; sectors: string[]; ministries: string[];
  risk_distribution: Record<'critical' | 'high' | 'medium' | 'low', number>;
  cost_exposure_by_risk_cr: Record<'critical' | 'high' | 'medium' | 'low', number>;
  predicted_cost_exposure_cr: number; model_version: string | null; dataset_snapshot: string | null; inference_timestamp: string;
}

export interface PortfolioRiskItem extends ProjectListItem {
  schedule_risk_probability: number; cost_risk_probability: number; estimated_schedule_extension_days: number;
  estimated_cost_escalation_pct: number; priority_score: number; priority_level: string; confidence: string; exposure_percentile: number;
  best_models: { schedule_classifier: string; cost_classifier: string; schedule_regressor: string; cost_regressor: string };
  schedule_drivers: Record<string, unknown>[]; cost_drivers: Record<string, unknown>[];
  observed: { schedule_extension_days: number | null; cost_escalation_pct: number | null; financial_progress_pct: number | null; physical_progress_pct: number | null };
}

export interface PortfolioSummaryResponse {
  projects: number; original_cost_cr: number; current_cost_basis_cr: number; expenditure_cr: number; predicted_cost_exposure_cr: number;
  risk_distribution: Record<'critical' | 'high' | 'medium' | 'low', number>;
  cost_exposure_by_risk_cr: Record<'critical' | 'high' | 'medium' | 'low', number>; sectors: number;
  dataset_snapshot: string | null; dataset_scope: string; model_version: string | null; model_scope: string | null; inference_timestamp: string;
  expenditure_progress: { project_code: string; physical_progress_pct: number; financial_progress_pct: number; group: 'On Track' | 'Monitor' | 'At Risk' }[];
  warning_drivers: { name: string; count: number }[]; risk_trend: null; risk_trend_status: string;
}

export interface PeerResponse {
  sector: string; peer_count: number;
  medians: { original_cost_cr: number | null; cost_escalation_pct: number | null; schedule_extension_days: number | null; financial_progress_pct: number | null; physical_progress_pct: number | null };
  peers: { project_code: string; project_name: string; original_cost_cr: number | null; cost_escalation_pct: number | null; schedule_extension_days: number | null }[];
}

export interface WarningResponse {
  available: boolean; reason: string | null; source: string | null;
  items: { date: string; type: string; severity: RiskLevel; message: string }[];
}

export interface LifecycleForecastResponse {
  project_id: string; project_name: string; model_version: string; snapshot_date: string; history_snapshots: number;
  predicted_cost_overrun_percentage: number; predicted_delay_days: number; risk_level: string; shap_explanation: ShapFactor[];
  explanation_scope: string; provenance: { run_id: string | null; dataset_fingerprint: string | null; verified: boolean }; model_scope: string;
}
