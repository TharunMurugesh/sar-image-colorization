import React from 'react';
import { X, CheckCircle, AlertCircle, Download, Clock, ShieldAlert, Cpu } from 'lucide-react';
import { ImageComparison } from './ImageComparison';
import { UncertaintyVisualizer } from './UncertaintyVisualizer';
import { resolveImageUrl } from '../services/api';

export function JobDetailsModal({ job, onClose }) {
  if (!job) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(3, 7, 18, 0.85)',
        backdropFilter: 'blur(10px)',
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        animation: 'fadeIn 0.2s ease-out'
      }}
      onClick={onClose}
    >
      <div
        className="glass-card"
        style={{
          maxWidth: '1000px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          padding: '24px',
          border: '1px solid var(--border-medium)',
          boxShadow: 'var(--shadow-lg)',
          position: 'relative'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              JOB ID: {job.id}
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {job.filename}
            </h3>
          </div>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onClose}
            style={{ borderRadius: '50%', width: '36px', height: '36px', padding: 0 }}
          >
            <X size={18} />
          </button>
        </div>

        {job.status === 'error' && (
          <div 
            style={{ 
              padding: '16px', 
              background: 'rgba(244, 63, 94, 0.1)', 
              border: '1px solid rgba(244, 63, 94, 0.3)', 
              borderRadius: 'var(--radius-md)',
              color: 'var(--rose-400)',
              marginBottom: '20px'
            }}
          >
            <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <AlertCircle size={18} />
              <span>Inference Failed</span>
            </div>
            <p style={{ fontSize: '0.86rem' }}>{job.error_message}</p>
          </div>
        )}

        {job.status === 'done' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="image-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
              <ImageComparison
                sarUrl={null}
                colorizedUrl={job.result_url}
                filename={job.filename}
              />
              <UncertaintyVisualizer
                uncertaintyUrl={job.uncertainty_url}
                uncertaintyMean={job.uncertainty_mean}
                filename={job.filename}
              />
            </div>

            <div 
              style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', 
                gap: '12px',
                background: 'rgba(0,0,0,0.3)',
                padding: '14px 18px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
                fontSize: '0.82rem'
              }}
            >
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Created At:</span>
                <div style={{ color: 'var(--text-primary)', marginTop: '2px' }}>
                  {new Date(job.created_at).toLocaleString()}
                </div>
              </div>

              <div>
                <span style={{ color: 'var(--text-muted)' }}>Spatial Extent:</span>
                <div style={{ color: 'var(--text-primary)', marginTop: '2px' }} className="font-mono">
                  {job.sar_width ? `${job.sar_width} &times; ${job.sar_height}` : '256 &times; 256 px'}
                </div>
              </div>

              <div>
                <span style={{ color: 'var(--text-muted)' }}>Channel Mode:</span>
                <div style={{ color: 'var(--text-primary)', marginTop: '2px' }} className="font-mono">
                  {job.sar_channels ? `${job.sar_channels} Bands` : '3 Bands'}
                </div>
              </div>

              <div>
                <span style={{ color: 'var(--text-muted)' }}>Mean Uncertainty:</span>
                <div style={{ color: 'var(--cyan-400)', marginTop: '2px', fontWeight: 600 }} className="font-mono">
                  {job.uncertainty_mean ? Number(job.uncertainty_mean).toFixed(6) : '—'}
                </div>
              </div>
            </div>
          </div>
        )}

        {(job.status === 'pending' || job.status === 'running') && (
          <div style={{ padding: '40px', textAlign: 'center' }}>
            <Clock size={36} className="spin" style={{ color: 'var(--cyan-400)', margin: '0 auto 12px' }} />
            <h4>Inference currently in progress...</h4>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem' }}>
              Executing Monte Carlo Dropout passes across SSG-U-Net decoder.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
