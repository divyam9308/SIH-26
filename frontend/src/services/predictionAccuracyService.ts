import { apiGet } from './api';

export interface ValidationReport {
  model_version: string;
  cost_model: { MAE: number; RMSE: number; R2: number; MAPE: number };
  delay_model: { MAE: number; MAE_days?: number; RMSE: number; R2: number; MAPE?: number };
  risk_model: { accuracy: number; precision?: number; macro_precision?: number; recall?: number; macro_recall?: number; f1?: number; macro_f1?: number };
  metadata: { training_start: number; training_end: number; test_start: number; test_end: number; testing_samples?: number; data_source?: string; validation_method?: string; features_used?: string[]; confidence_calibration_status?: string };
}
export interface ValidationRow {
  project_id: string | null; project_name: string; sector: string; predicted_cost_overrun: number; actual_cost_overrun: number;
  cost_error: number; predicted_delay_days: number; actual_delay_days: number; delay_error: number;
  predicted_risk: number | string; actual_risk: number | string; risk_probability: number | null; model_confidence_percentage: number | null;
}
export interface RollingFold { test_year: number; cost_MAE: number; delay_MAE_days: number; risk_f1: number }
export interface ModelImportance { feature: string; importance: number }
export interface PredictionAccuracyData {
  report: ValidationReport; rows: ValidationRow[]; total: number;
  rolling: { model_version: string; folds: RollingFold[]; fold_count: number; policy?: string };
  importance: { model_version: string; cost_model: ModelImportance[]; delay_model: ModelImportance[]; risk_model: ModelImportance[] };
}

export async function getPredictionAccuracyData(signal?: AbortSignal): Promise<PredictionAccuracyData> {
  const [report, evidence, rolling, importance] = await Promise.all([
    apiGet<ValidationReport>('/api/models/validation', signal),
    apiGet<{ model_version: string; items: ValidationRow[]; total: number }>('/api/models/prediction-validation?limit=500', signal),
    apiGet<PredictionAccuracyData['rolling']>('/api/models/rolling-validation', signal),
    apiGet<PredictionAccuracyData['importance']>('/api/models/importance', signal),
  ]);
  return { report, rows: evidence.items, total: evidence.total, rolling, importance };
}
