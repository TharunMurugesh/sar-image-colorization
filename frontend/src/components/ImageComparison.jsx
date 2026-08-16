import React, { useState, useRef, useEffect } from 'react';
import { Columns, Eye, Download, Maximize2, Split } from 'lucide-react';
import { resolveImageUrl } from '../services/api';

export function ImageComparison({ sarUrl, colorizedUrl, filename }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'side-by-side'
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  const fullColorUrl = resolveImageUrl(colorizedUrl);
  const fullSarUrl = sarUrl; // Can be a local object URL or remote URL

  const handleMouseDown = () => setIsDragging(true);
  const handleTouchStart = () => setIsDragging(true);

  useEffect(() => {
    const handleMouseUp = () => setIsDragging(false);
    const handleTouchEnd = () => setIsDragging(false);

    const handleMove = (clientX) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const position = (x / rect.width) * 100;
      setSliderPosition(Math.max(0, Math.min(100, position)));
    };

    const handleMouseMove = (e) => handleMove(e.clientX);
    const handleTouchMove = (e) => {
      if (e.touches && e.touches[0]) {
        handleMove(e.touches[0].clientX);
      }
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleTouchEnd);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
    };
  }, [isDragging]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h4 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Colorization Comparison View
        </h4>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            type="button"
            className={`btn btn-sm ${viewMode === 'split' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setViewMode('split')}
            title="Interactive Split Slider"
          >
            <Split size={14} />
            <span>Split Slider</span>
          </button>
          
          <button
            type="button"
            className={`btn btn-sm ${viewMode === 'side-by-side' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setViewMode('side-by-side')}
            title="Side-by-Side Gallery"
          >
            <Columns size={14} />
            <span>Side by Side</span>
          </button>

          {fullColorUrl && (
            <a
              href={fullColorUrl}
              download={`${filename || 'colorized'}_colorized.png`}
              target="_blank"
              rel="noreferrer"
              className="btn btn-secondary btn-sm"
              title="Download Colorized PNG"
            >
              <Download size={14} />
              <span>Download</span>
            </a>
          )}
        </div>
      </div>

      {viewMode === 'split' ? (
        <div className="comparison-container" ref={containerRef}>
          {/* Base Image (Colorized RGB) */}
          <img
            src={fullColorUrl}
            alt="Colorized Output"
            className="comparison-image"
          />
          <div className="comparison-label label-color">SSG-U-Net Colorized</div>

          {/* Clipped Top Image (SAR Input) */}
          {fullSarUrl && (
            <div
              className="comparison-clip"
              style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}
            >
              <img
                src={fullSarUrl}
                alt="Input SAR"
                className="comparison-image"
              />
              <div className="comparison-label label-sar">Input SAR</div>
            </div>
          )}

          {/* Draggable Divider Handle */}
          {fullSarUrl && (
            <div
              className="comparison-slider-handle"
              style={{ left: `${sliderPosition}%` }}
              onMouseDown={handleMouseDown}
              onTouchStart={handleTouchStart}
            >
              <div className="handle-badge">
                <Split size={16} />
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="image-grid" style={{ marginTop: '0' }}>
          {fullSarUrl && (
            <div className="image-card">
              <div className="image-card-header">
                <h4>Input SAR</h4>
                <span className="format-tag">MONO / POL</span>
              </div>
              <div className="image-card-body">
                <img src={fullSarUrl} alt="Input SAR" />
              </div>
            </div>
          )}

          <div className="image-card">
            <div className="image-card-header">
              <h4>SSG-U-Net Colorized</h4>
              <span className="badge badge-done">RGB OUTPUT</span>
            </div>
            <div className="image-card-body">
              <img src={fullColorUrl} alt="Colorized Output" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
