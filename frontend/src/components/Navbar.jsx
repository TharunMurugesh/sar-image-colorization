import React from 'react';
import { Satellite, History, Sparkles, Cpu, Activity } from 'lucide-react';

export function Navbar({ activeTab, setActiveTab, health }) {
  const isHealthy = health?.status === 'ok';
  const isDegraded = health?.status === 'degraded';
  const isOffline = health?.status === 'offline';

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="brand-badge" onClick={() => setActiveTab('colorize')}>
          <div className="satellite-icon-ring">
            <Satellite size={22} />
          </div>
          <div className="brand-info">
            <div className="brand-tag">ISRO SIH1733 &bull; SAR AI</div>
            <h1>SAR Image Colorization</h1>
          </div>
        </div>

        <nav className="nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === 'colorize' ? 'active' : ''}`}
            onClick={() => setActiveTab('colorize')}
          >
            <Sparkles size={16} />
            <span>Colorize</span>
          </button>
          
          <button
            className={`nav-tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={16} />
            <span>History</span>
          </button>

          <button
            className={`nav-tab-btn ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            <Cpu size={16} />
            <span>Architecture</span>
          </button>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div 
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px', 
              background: 'rgba(15, 23, 42, 0.6)', 
              padding: '6px 12px', 
              borderRadius: 'var(--radius-full)',
              border: '1px solid var(--border-subtle)',
              fontSize: '0.78rem',
              fontFamily: 'var(--font-mono)'
            }}
            title={
              isHealthy
                ? 'Backend and SSG-U-Net model are operational'
                : isDegraded
                ? 'Backend connected; model checkpoint pending'
                : 'Backend is offline or unreachable'
            }
          >
            <div 
              className={`pulse-dot ${
                isHealthy ? 'dot-emerald' : isDegraded ? 'dot-amber' : 'dot-rose'
              }`}
            />
            <span style={{ color: 'var(--text-secondary)' }}>
              {isHealthy ? 'MODEL READY' : isDegraded ? 'CHECKPOINT PENDING' : 'BACKEND OFFLINE'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
