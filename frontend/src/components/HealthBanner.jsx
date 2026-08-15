import React from 'react';
import { AlertTriangle, CheckCircle2, XCircle, Info, RefreshCw } from 'lucide-react';

export function HealthBanner({ health, loading, onRefresh }) {
  if (loading && !health) {
    return null;
  }

  const status = health?.status || 'offline';

  if (status === 'ok') {
    return (
      <div className="health-banner status-ok">
        <div className="health-status-content">
          <CheckCircle2 size={20} color="var(--emerald-400)" />
          <div>
            <strong>Operational:</strong> Trained SSG-U-Net model checkpoint loaded. Ready for SAR colorization.
          </div>
        </div>
        <button 
          className="btn btn-secondary btn-sm" 
          onClick={onRefresh} 
          title="Refresh health status"
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          <span>Status</span>
        </button>
      </div>
    );
  }

  if (status === 'degraded') {
    return (
      <div className="health-banner status-degraded">
        <div className="health-status-content">
          <AlertTriangle size={20} color="var(--amber-400)" style={{ flexShrink: 0 }} />
          <div>
            <strong>Model Checkpoint Unavailable:</strong> The FastAPI backend is online, but no trained weights were found at{' '}
            <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px' }}>
              {health.checkpoint_path || 'runtime/checkpoints/best_model.pt'}
            </code>.
            <div style={{ marginTop: '4px', fontSize: '0.8rem', opacity: 0.9 }}>
              Uploads will return <code>HTTP 503</code> until the Day 5 training pipeline completes and produces the model checkpoint. Fake results or random weights are strictly disabled.
            </div>
          </div>
        </div>
        <button 
          className="btn btn-secondary btn-sm" 
          onClick={onRefresh} 
          title="Check again"
          style={{ flexShrink: 0 }}
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>
    );
  }

  return (
    <div className="health-banner status-offline">
      <div className="health-status-content">
        <XCircle size={20} color="var(--rose-400)" style={{ flexShrink: 0 }} />
        <div>
          <strong>Backend Offline:</strong> Unable to connect to the FastAPI server at{' '}
          <code>{import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}</code>.
          <div style={{ marginTop: '4px', fontSize: '0.8rem', opacity: 0.9 }}>
            Please ensure the backend is running via <code>uvicorn backend.app.main:app --port 8000</code>.
          </div>
        </div>
      </div>
      <button 
        className="btn btn-danger btn-sm" 
        onClick={onRefresh}
        style={{ flexShrink: 0 }}
      >
        <RefreshCw size={14} className={loading ? 'spin' : ''} />
        <span>Retry</span>
      </button>
    </div>
  );
}
