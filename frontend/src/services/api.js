/**
 * services/api.js
 * Centralized API client for the SAR Colorization FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Format relative result URL into full backend URL if needed.
 */
export function resolveImageUrl(url) {
  if (!url) return null;
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }
  const cleanBase = API_BASE_URL.replace(/\/+$/, '');
  const cleanPath = url.startsWith('/') ? url : `/${url}`;
  return `${cleanBase}${cleanPath}`;
}

/**
 * Check system & model health.
 * GET /api/health
 */
export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) {
      throw new Error(`Health check failed with HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    return {
      status: 'offline',
      model_loaded: false,
      checkpoint_exists: false,
      gpu_available: false,
      error: err.message || 'Cannot reach backend server',
    };
  }
}

/**
 * Upload a SAR image and initiate colorization job.
 * POST /api/colorize
 * @param {File} file
 */
export async function uploadColorizeImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/api/colorize`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const errorMsg = data.detail || `Upload failed (HTTP ${res.status})`;
    const error = new Error(errorMsg);
    error.status = res.status;
    error.detail = data.detail;
    throw error;
  }

  return data;
}

/**
 * Poll job status and fetch results.
 * GET /api/colorize/{job_id}
 * @param {string} jobId
 */
export async function fetchJobStatus(jobId) {
  const res = await fetch(`${API_BASE_URL}/api/colorize/${jobId}`);
  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const errorMsg = data.detail || `Failed to fetch job (HTTP ${res.status})`;
    const error = new Error(errorMsg);
    error.status = res.status;
    throw error;
  }

  return data;
}

/**
 * Fetch paginated history of colorization jobs.
 * GET /api/history
 * @param {Object} options - { limit, offset, status }
 */
export async function fetchHistory({ limit = 20, offset = 0, status = null } = {}) {
  const params = new URLSearchParams();
  if (limit) params.append('limit', limit);
  if (offset) params.append('offset', offset);
  if (status) params.append('status', status);

  const url = `${API_BASE_URL}/api/history${params.toString() ? `?${params.toString()}` : ''}`;
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const errorMsg = data.detail || `Failed to fetch history (HTTP ${res.status})`;
    const error = new Error(errorMsg);
    error.status = res.status;
    throw error;
  }

  return data;
}
