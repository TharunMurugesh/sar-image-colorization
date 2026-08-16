import React, { useState, useRef } from 'react';
import { UploadCloud, FileImage, X, AlertCircle, FileCode } from 'lucide-react';

const ACCEPTED_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
const ACCEPT_STRING = '.tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg';

export function Dropzone({ selectedFile, onFileSelect, onClearFile, disabled }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [filePreview, setFilePreview] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = (file) => {
    setError(null);
    if (!file) return;

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Invalid file format: ${ext}. Only ${ACCEPTED_EXTENSIONS.join(', ')} are supported.`);
      return;
    }

    // Generate local preview URL for web-friendly image types (PNG/JPG)
    if (file.type.startsWith('image/') && !ext.includes('tif')) {
      const url = URL.createObjectURL(file);
      setFilePreview(url);
    } else {
      setFilePreview(null);
    }

    onFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    if (filePreview) URL.revokeObjectURL(filePreview);
    setFilePreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    onClearFile();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPT_STRING}
        style={{ display: 'none' }}
        onChange={handleFileInputChange}
        disabled={disabled}
      />

      {!selectedFile ? (
        <div
          className={`dropzone ${isDragOver ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <div className="dropzone-icon-wrapper">
            <UploadCloud size={32} />
          </div>
          <h3 className="dropzone-title">Upload SAR Target Image</h3>
          <p className="dropzone-desc">
            Drag & drop your single/dual-polarization SAR imagery or click to browse
          </p>

          <div className="file-format-tags">
            <span className="format-tag">GEOTIFF (.tif / .tiff)</span>
            <span className="format-tag">PNG</span>
            <span className="format-tag">JPG / JPEG</span>
          </div>
        </div>
      ) : (
        <div className="file-preview-card">
          <div className="file-info-group">
            {filePreview ? (
              <img src={filePreview} alt="Selected preview" className="file-thumb-mini" />
            ) : (
              <div className="file-thumb-mini">
                <FileCode size={24} />
              </div>
            )}
            <div className="file-details">
              <h4>{selectedFile.name}</h4>
              <div className="file-meta">
                <span>{formatFileSize(selectedFile.size)}</span>
                <span>&bull;</span>
                <span style={{ textTransform: 'uppercase' }}>
                  {selectedFile.name.split('.').pop()} FILE
                </span>
                {selectedFile.name.toLowerCase().includes('tif') && (
                  <>
                    <span>&bull;</span>
                    <span style={{ color: 'var(--cyan-400)' }}>16-bit / 32-bit Float GeoTIFF</span>
                  </>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
            >
              Replace
            </button>
            <button
              type="button"
              className="btn btn-danger btn-sm"
              onClick={handleClear}
              disabled={disabled}
              title="Clear selected file"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {error && (
        <div 
          style={{ 
            marginTop: '12px', 
            padding: '10px 14px', 
            background: 'rgba(244, 63, 94, 0.1)', 
            border: '1px solid rgba(244, 63, 94, 0.25)', 
            borderRadius: 'var(--radius-sm)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: 'var(--rose-400)',
            fontSize: '0.84rem'
          }}
        >
          <AlertCircle size={16} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
