import { useCallback, useEffect, useState } from 'react';
import { getDashboardData } from '../services/dashboardService';
import type { DashboardData } from '../types/dashboard';

export function useDashboardData(window = '2001_2017') {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const refresh = useCallback(() => setRefreshKey((value) => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(null);
    getDashboardData(window, controller.signal).then(setData).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === 'AbortError') return;
      setError(reason instanceof Error ? reason.message : 'Dashboard data is unavailable.');
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [refreshKey, window]);
  return { data, loading, error, refresh };
}
