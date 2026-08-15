import React from 'react';
import { Clock, Eye, AlertCircle, CheckCircle, RefreshCw, FileText } from 'lucide-react';
import { resolveImageUrl } from '../services/api';

export function HistoryTable({ jobs, loading, onRefresh, onViewJob }) {
  const formatDate = (isoString) => {
    if (!isoString) return '—';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const renderStatusBadge = (status) => {
    switch (status) {
      case 'done':
        return (
          <span className="badge badge-done">
            <CheckCircle size={12} />
            <span>Done</span>
          </span>
        );
      case 'running':
        return (
          <span className="badge badge-running">
            <RefreshCw size={12} className="spin" />
            <span>Running</span>
          </span>
        );
      case 'pending':
        return (
          <span className="badge badge-pending">
            <Clock size={12} />
            <span>Pending</span>
          </span>
        );
      case 'error':
        return (
          <span className="badge badge-error">
            <AlertCircle size={12} />
            <span>Error</span>
          </span>
        );
      default:
        return <span className="badge">{status}</span>;
    }
  };

  if (!loading && (!jobs || jobs.length === 0)) {
    return (
      <div 
        className="glass-card" 
        style={{ 
          padding: '60px 24px', 
          textAlign: 'center',
          color: 'var(--text-muted)'
        }}
      >
        <FileText size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
        <h4 style={{ fontSize: '1.1rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
          No Colorization History Found
        </h4>
        <p style={{ fontSize: '0.86rem', maxWidth: '400px', margin: '0 auto 20px' }}>
          Jobs created during SAR image processing will be automatically recorded here with outputs and uncertainty diagnostics.
        </p>
        <button className="btn btn-secondary btn-sm" onClick={onRefresh}>
          <RefreshCw size={14} />
          <span>Check Again</span>
        </button>
      </div>
    );
  }

  return (
    <div className="table-container glass-card">
      <table className="custom-table">
        <thead>
          <tr>
            <th>Job ID</th>
            <th>Filename</th>
            <th>Timestamp</th>
            <th>Resolution</th>
            <th>Mean Variance</th>
            <th>Status</th>
            <th style={{ textAlign: 'right' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {job.id.slice(0, 8)}...
              </td>
              <td>
                <strong style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                  {job.filename}
                </strong>
              </td>
              <td style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                {formatDate(job.created_at)}
              </td>
              <td className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {job.sar_width && job.sar_height 
                  ? `${job.sar_width} &times; ${job.sar_height} (${job.sar_channels}ch)`
                  : '256 &times; 256'
                }
              </td>
              <td className="font-mono" style={{ fontSize: '0.8rem' }}>
                {job.uncertainty_mean !== null && job.uncertainty_mean !== undefined
                  ? Number(job.uncertainty_mean).toFixed(5)
                  : '—'}
              </td>
              <td>{renderStatusBadge(job.status)}</td>
              <td style={{ textAlign: 'right' }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => onViewJob(job)}
                  title="View job diagnostics and images"
                >
                  <Eye size={14} />
                  <span>Inspect</span>
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
