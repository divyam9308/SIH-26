const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE_URL = configuredBaseUrl ? configuredBaseUrl.replace(/\/$/, '') : '';

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) { super(message); }
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { signal, headers: { Accept: 'application/json' } });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ApiError(0, 'Backend unavailable. Start the FastAPI service and try again.');
  }
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try { const body = await response.json() as { detail?: string }; if (body.detail) message = body.detail; } catch { /* non-JSON error */ }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}
