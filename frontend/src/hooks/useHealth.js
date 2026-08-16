import { useState, useEffect, useCallback } from 'react';
import { fetchHealth } from '../services/api';

/**
 * Custom hook to monitor backend & model health status.
 * Auto-refreshes every intervalMs (default 15s).
 */
export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastChecked, setLastChecked] = useState(null);

  const checkHealth = useCallback(async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
      setLastChecked(new Date());
    } catch (err) {
      setHealth({
        status: 'offline',
        model_loaded: false,
        checkpoint_exists: false,
        gpu_available: false,
        error: err.message,
      });
      setLastChecked(new Date());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, intervalMs);
    return () => clearInterval(interval);
  }, [checkHealth, intervalMs]);

  return { health, loading, lastChecked, refetchHealth: checkHealth };
}
