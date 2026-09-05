import { apiGet } from './api';

export type TrainingWindowMetric = {
  start_year: number;
  end_year: number;
  cost_mae: number | null;
  delay_mae_days: number | null;
  cost_r2: number | null;
  delay_r2: number | null;
  sample_count: number;
};

export type TrainingWindowPerformance = {
  windows: TrainingWindowMetric[];
  evaluation_period: string;
  sample_count: number;
  generated_at: string;
  methodology: string;
};

export function getTrainingWindowPerformance(signal?: AbortSignal) {
  return apiGet<TrainingWindowPerformance>('/api/models/training-window-performance', signal);
}
