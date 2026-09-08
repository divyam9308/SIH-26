import { apiGet } from './api';

export interface ValidationReport {
  model_version: string;
  cost_model: { MAE?: number; RMSE?: number; R2?: number; MAPE?: number };
  delay_model: { MAE?: number; MAE_days?: number; RMSE?: number; R2?: number; MAPE?: number };
  risk_model: { accuracy?: number; precision?: number; macro_precision?: number; recall?: number; macro_recall?: number; f1?: number; macro_f1?: number };
  metadata: {
    training_start?: number; training_end?: number; test_start?: number; test_end?: number; testing_samples?: number;
    training_projects?: number; evaluation_projects?: number; training_snapshots?: number; test_snapshots?: number;
    unique_training_projects?: number; unique_test_projects?: number; data_source?: string; validation_method?: string;
    features_used?: string[]; confidence_calibration_status?: string;
    feature_quality?: { data_quality_score?: number; as_of_evidence_coverage?: number };
  };
}
export interface ValidationRow {
  project_id: string | null; project_name?: string; sector?: string; predicted_cost_overrun: number | null; actual_cost_overrun: number | null;
  cost_error: number | null; predicted_delay_days: number | null; actual_delay_days: number | null; delay_error: number | null;
  predicted_risk?: number | string; actual_risk?: number | string; risk_probability?: number | null; model_confidence_percentage: number | null;
}
export interface RollingFold { test_year: number; cost_MAE: number; delay_MAE_days: number; risk_f1?: number; test_projects?: number; test_snapshots?: number }
export interface RollingValidation { model_version: string; folds: RollingFold[]; fold_count: number; policy?: string; status?: string; training_period?: number[]; testing_period?: number[] }
export interface PredictionAccuracyData { report: ValidationReport; rows: ValidationRow[]; total: number; rolling: RollingValidation; }

export async function getPredictionAccuracyData(window: string, signal?: AbortSignal): Promise<PredictionAccuracyData> {
  const query = `?model_version=${encodeURIComponent(window)}`;
  const [report, evidence, rolling] = await Promise.all([
    apiGet<ValidationReport>(`/api/models/validation${query}`, signal),
    apiGet<{ model_version?: string; items: ValidationRow[]; total: number }>(`/api/models/prediction-validation?limit=500&model_version=${encodeURIComponent(window)}&completion_year_start=2023&completion_year_end=2025`, signal),
    apiGet<RollingValidation>(`/api/models/rolling-validation${query}`, signal),
  ]);

  if (!evidence.items?.length || evidence.total < 1) {
    throw new Error(`Production validation evidence for ${window} is empty.`);
  }
  if (!rolling.folds?.length || rolling.fold_count !== rolling.folds.length) {
    throw new Error(`Annual holdout metrics for ${window} are missing or incomplete.`);
  }
  const visibleRolling = { ...rolling, folds: rolling.folds.filter((fold) => Number(fold.test_year) >= 2023) };
  visibleRolling.fold_count = visibleRolling.folds.length;
  const years = visibleRolling.folds.map((fold) => Number(fold.test_year));
  if (window === '2001_2021' && JSON.stringify(years) !== JSON.stringify([2023, 2024, 2025])) {
    throw new Error(`Expected 2022–2025 graph evidence for ${window}; received ${years.join(', ')}.`);
  }

  return { report, rows: evidence.items, total: evidence.total, rolling: visibleRolling };
}
