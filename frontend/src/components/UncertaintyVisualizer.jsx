import React from 'react';
import { ShieldAlert, Download, Info, BarChart2 } from 'lucide-react';
import { resolveImageUrl } from '../services/api';

export function UncertaintyVisualizer({ uncertaintyUrl, uncertaintyMean, filename }) {
  const fullUncertaintyUrl = resolveImageUrl(uncertaintyUrl);

  // Derive qualitative trust level based on uncertainty mean
  let trustBadge = { label: 'Optimal Confidence', color: 'var(--emerald-400)', bg: 'rgba(16, 185, 129, 0.15)' };
  if (uncertaintyMean > 0.08) {
    trustBadge = { label: 'High Uncertainty (Attenuated)', color: 'var(--rose-400)', bg: 'rgba(244, 63, 94, 0.15)' };
  } else if (uncertaintyMean > 0.03) {
    trustBadge = { label: 'Moderate Confidence', color: 'var(--amber-400)', bg: 'rgba(245, 158, 11, 0.15)' };
  }

  return (
    <div className="image-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="image-card-header">
        <h4>
          <ShieldAlert size={16} color="var(--amber-400)" />
          <span>MC-Dropout Uncertainty Map</span>
        </h4>
        {fullUncertaintyUrl && (
          <a
            href={fullUncertaintyUrl}
            download={`${filename || 'colorized'}_uncertainty.png`}
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary btn-sm"
            title="Download Heatmap PNG"
          >
            <Download size={14} />
            <span>Heatmap</span>
          </a>
        )}
      </div>

      <div className="image-card-body" style={{ flex: 1 }}>
        {fullUncertaintyUrl ? (
          <img src={fullUncertaintyUrl} alt="MC-Dropout Uncertainty Heatmap" />
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            No uncertainty map available
          </div>
        )}
      </div>

      <div className="uncertainty-legend">
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: 'var(--cyan-400)' }}>Certain</span>
          <span style={{ fontSize: '0.7rem' }}>(Low &sigma;&sup2;)</span>
        </div>
        <div className="thermal-bar" title="Thermal variance distribution across MC passes" />
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: 'var(--rose-400)' }}>Uncertain</span>
          <span style={{ fontSize: '0.7rem' }}>(High &sigma;&sup2;)</span>
        </div>
      </div>

      {uncertaintyMean !== null && uncertaintyMean !== undefined && (
        <div 
          style={{ 
            padding: '12px 18px', 
            background: 'rgba(15, 23, 42, 0.95)', 
            borderTop: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.82rem'
          }}
        >
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Mean Pixel Variance: </span>
            <strong className="font-mono" style={{ color: 'var(--text-primary)' }}>
              {Number(uncertaintyMean).toFixed(6)}
            </strong>
          </div>

          <div
            style={{
              padding: '2px 8px',
              borderRadius: 'var(--radius-full)',
              background: trustBadge.bg,
              color: trustBadge.color,
              fontWeight: 600,
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)'
            }}
          >
            {trustBadge.label}
          </div>
        </div>
      )}
    </div>
  );
}
