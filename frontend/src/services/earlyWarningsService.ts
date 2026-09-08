import { apiGet } from './api';
import type { EarlyWarningsResponse, WarningStatus } from '../types/earlyWarnings';

export interface EarlyWarningsQuery { window?: string; projectId?: string; asOf?: string; search?: string; severity?: WarningStatus | 'All'; sector?: string; ministry?: string; thresholdOnly?: boolean; }

export function getEarlyWarnings(query: EarlyWarningsQuery, signal?: AbortSignal) {
  const params = new URLSearchParams({ window: query.window ?? '2001_2022' });
  if (query.projectId) params.set('project_id', query.projectId);
  if (query.asOf) params.set('as_of', query.asOf);
  if (query.search) params.set('search', query.search);
  if (query.severity && query.severity !== 'All') params.set('severity', query.severity);
  if (query.sector && query.sector !== 'All Sectors') params.set('sector', query.sector);
  if (query.ministry && query.ministry !== 'All Ministries') params.set('ministry', query.ministry);
  if (query.thresholdOnly) params.set('threshold_only', 'true');
  return apiGet<EarlyWarningsResponse>(`/api/early-warnings?${params}`, signal);
}
