import React, { useState, useEffect } from 'react';
import { History, RefreshCw, Filter } from 'lucide-react';
import { HistoryTable } from '../components/HistoryTable';
import { JobDetailsModal } from '../components/JobDetailsModal';
import { fetchHistory } from '../services/api';

export function HistoryPage() {
  const [jobs, setJobs] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [error, setError] = useState(null);

  const loadHistory = async (status = statusFilter) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHistory({ limit: 50, status });
      setJobs(data.jobs || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message || 'Failed to load colorization history.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory(statusFilter);
  }, [statusFilter]);

  const handleFilterChange = (status) => {
    setStatusFilter(status);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Colorization History Archive
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Historical record of all synthesized SAR transformations, uncertainty maps, and execution diagnostics.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="nav-tabs" style={{ padding: '3px' }}>
            <button
              className={`nav-tab-btn ${statusFilter === null ? 'active' : ''}`}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
              onClick={() => handleFilterChange(null)}
            >
              All ({total})
            </button>
            <button
              className={`nav-tab-btn ${statusFilter === 'done' ? 'active' : ''}`}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
              onClick={() => handleFilterChange('done')}
            >
              Completed
            </button>
            <button
              className={`nav-tab-btn ${statusFilter === 'error' ? 'active' : ''}`}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
              onClick={() => handleFilterChange('error')}
            >
              Failed
            </button>
          </div>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => loadHistory(statusFilter)}
            title="Refresh history table"
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div 
          style={{ 
            padding: '12px 18px', 
            background: 'rgba(244, 63, 94, 0.1)', 
            border: '1px solid rgba(244, 63, 94, 0.3)', 
            borderRadius: 'var(--radius-md)',
            color: 'var(--rose-400)',
            fontSize: '0.86rem'
          }}
        >
          {error}
        </div>
      )}

      <HistoryTable
        jobs={jobs}
        loading={loading}
        onRefresh={() => loadHistory(statusFilter)}
        onViewJob={(job) => setSelectedJob(job)}
      />

      {/* Modal for detailed job inspection */}
      {selectedJob && (
        <JobDetailsModal
          job={selectedJob}
          onClose={() => setSelectedJob(null)}
        />
      )}
    </div>
  );
}
