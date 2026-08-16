import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, AlertCircle, RefreshCw, Layers, CheckCircle2, ArrowRight } from 'lucide-react';
import { Dropzone } from '../components/Dropzone';
import { ImageComparison } from '../components/ImageComparison';
import { UncertaintyVisualizer } from '../components/UncertaintyVisualizer';
import { CaveatsNotice } from '../components/CaveatsNotice';
import { uploadColorizeImage, fetchJobStatus } from '../services/api';

export function ColorizePage({ health, onJobCompleted }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [localSarUrl, setLocalSarUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [currentJob, setCurrentJob] = useState(null);
  const [error, setError] = useState(null);

  const pollIntervalRef = useRef(null);

  const isModelReady = health?.checkpoint_exists && health?.status === 'ok';

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setError(null);
    setCurrentJob(null);

    // Create local object URL for preview if it's an image
    if (file && file.type.startsWith('image/') && !file.name.toLowerCase().includes('tif')) {
      const url = URL.createObjectURL(file);
      setLocalSarUrl(url);
    } else {
      setLocalSarUrl(null);
    }
  };

  const handleClearFile = () => {
    if (localSarUrl) URL.revokeObjectURL(localSarUrl);
    setSelectedFile(null);
    setLocalSarUrl(null);
    setError(null);
    setCurrentJob(null);
  };

  // Polling loop for background job
  const startPolling = (jobId) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const job = await fetchJobStatus(jobId);
        setCurrentJob(job);

        if (job.status === 'done') {
          clearInterval(pollIntervalRef.current);
          setLoading(false);
          setStatusMessage('Colorization complete!');
          if (onJobCompleted) onJobCompleted(job);
        } else if (job.status === 'error') {
          clearInterval(pollIntervalRef.current);
          setLoading(false);
          setError(job.error_message || 'Inference process encountered an error.');
        } else if (job.status === 'running') {
          setStatusMessage('Executing MC-Dropout passes (10 stochastic samples)...');
        } else {
          setStatusMessage('Job enqueued in background worker...');
        }
      } catch (err) {
        clearInterval(pollIntervalRef.current);
        setLoading(false);
        setError(err.message || 'Failed to poll job status.');
      }
    }, 1200);
  };

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile || loading) return;

    setError(null);
    setLoading(true);
    setStatusMessage('Uploading SAR target to inference pipeline...');

    try {
      const initialJob = await uploadColorizeImage(selectedFile);
      setCurrentJob(initialJob);
      setStatusMessage('Analyzing structural gradients...');
      startPolling(initialJob.id);
    } catch (err) {
      setLoading(false);
      setError(
        err.detail || 
        err.message || 
        'Upload failed. Please check backend connection.'
      );
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Upload & Action Card */}
      <div className="glass-card" style={{ padding: '28px' }}>
        <div style={{ marginBottom: '20px' }}>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            SAR-to-Optical Reconstruction Studio
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Upload raw Synthetic Aperture Radar imagery to synthesize physically-informed optical representations with Bayesian uncertainty estimation.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <Dropzone
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            onClearFile={handleClearFile}
            disabled={loading}
          />

          <div style={{ marginTop: '22px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
              <Layers size={16} />
              <span>Pipeline: ResNet18 + Sobel Guidance &bull; 256&times;256 Patching &bull; MC-Dropout (M=10)</span>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={!selectedFile || loading}
              style={{ minWidth: '180px' }}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Synthesize Color</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Loading Progress State */}
        {loading && (
          <div 
            style={{ 
              marginTop: '20px', 
              padding: '16px 20px', 
              background: 'rgba(0, 210, 255, 0.08)', 
              border: '1px solid rgba(0, 210, 255, 0.25)', 
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              alignItems: 'center',
              gap: '14px',
              color: 'var(--cyan-400)'
            }}
          >
            <RefreshCw size={20} className="spin" style={{ flexShrink: 0 }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.92rem' }}>Processing SAR Tensor</div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                {statusMessage}
              </div>
            </div>
          </div>
        )}

        {/* Error Callout */}
        {error && (
          <div 
            style={{ 
              marginTop: '20px', 
              padding: '16px 20px', 
              background: 'rgba(244, 63, 94, 0.1)', 
              border: '1px solid rgba(244, 63, 94, 0.3)', 
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '14px',
              color: 'var(--rose-400)'
            }}
          >
            <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.92rem' }}>Colorization Request Failed</div>
              <p style={{ fontSize: '0.84rem', marginTop: '4px', color: '#fecdd3', lineHeight: 1.45 }}>
                {error}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Results Display Area */}
      {currentJob && currentJob.status === 'done' && (
        <div className="glass-card" style={{ padding: '28px', animation: 'fadeIn 0.3s ease-out' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <CheckCircle2 size={22} color="var(--emerald-400)" />
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Optical Reconstruction Results
                </h3>
                <span className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  File: {currentJob.filename} &bull; Output Size: {currentJob.sar_width || 256}&times;{currentJob.sar_height || 256}
                </span>
              </div>
            </div>

            <div className="badge badge-done">
              SSG-U-Net Output Ready
            </div>
          </div>

          <div 
            style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', 
              gap: '24px',
              alignItems: 'stretch'
            }}
          >
            {/* Left Column: Image Comparison */}
            <div>
              <ImageComparison
                sarUrl={localSarUrl}
                colorizedUrl={currentJob.result_url}
                filename={currentJob.filename}
              />
            </div>

            {/* Right Column: Uncertainty Visualizer */}
            <div>
              <UncertaintyVisualizer
                uncertaintyUrl={currentJob.uncertainty_url}
                uncertaintyMean={currentJob.uncertainty_mean}
                filename={currentJob.filename}
              />
            </div>
          </div>
        </div>
      )}

      {/* Caveats and Physical Disclaimers */}
      <CaveatsNotice />
    </div>
  );
}
