import { apiGet } from './api';

export interface ValidationReport {
  model_version: string;
  cost_model: { MAE?: number; RMSE?: number; R2?: number; MAPE?: number };
  delay_model: { MAE?: number; MAE_days?: number; RMSE?: number; R2?: number; MAPE?: number };
  risk_model: { accuracy?: number; precision?: number; macro_precision?: number; recall?: number; macro_recall?: number; f1?: number; macro_f1?: number };
  metadata: {
    training_start?: number; training_end?: number; test_start?: number; test_end?: number;
    training_projects?: number; evaluation_projects?: number; training_snapshots?: number; test_snapshots?: number;
    unique_training_projects?: number; unique_test_projects?: number; data_source?: string; validation_method?: string;
    feature_quality?: { data_quality_score?: number; as_of_evidence_coverage?: number };
  };
}
export interface ValidationRow {
  project_id: string | null; project_name?: string; predicted_cost_overrun: number | null; actual_cost_overrun: number | null;
  cost_error: number | null; predicted_delay_days: number | null; actual_delay_days: number | null; delay_error: number | null;
  model_confidence_percentage: number | null;
}
export interface RollingFold { test_year: number; cost_MAE: number; delay_MAE_days: number; risk_f1?: number }
export interface RollingValidation { model_version: string; folds: RollingFold[]; fold_count: number; policy?: string; status?: string }
export interface PredictionAccuracyData { report: ValidationReport; rows: ValidationRow[] | null; total: number | null; rolling: RollingValidation | null; }

const isAbort = (reason: unknown) => reason instanceof DOMException && reason.name === 'AbortError';

export async function getPredictionAccuracyData(window: string, signal?: AbortSignal): Promise<PredictionAccuracyData> {
  const query = `?model_version=${encodeURIComponent(window)}`;
  const report = await apiGet<ValidationReport>(`/api/models/validation${query}`, signal);
  const [evidence, rolling] = await Promise.all([
    apiGet<{ items: ValidationRow[]; total: number }>(`/api/models/prediction-validation?limit=500&model_version=${encodeURIComponent(window)}`, signal).catch((reason: unknown) => { if (isAbort(reason)) throw reason; return null; }),
    apiGet<RollingValidation>(`/api/models/rolling-validation${query}`, signal).catch((reason: unknown) => { if (isAbort(reason)) throw reason; return null; }),
  ]);
  return { report, rows: evidence?.items ?? null, total: evidence?.total ?? null, rolling };
}
