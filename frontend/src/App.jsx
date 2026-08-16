import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { HealthBanner } from './components/HealthBanner';
import { ColorizePage } from './pages/ColorizePage';
import { HistoryPage } from './pages/HistoryPage';
import { AboutPage } from './pages/AboutPage';
import { useHealth } from './hooks/useHealth';

export function App() {
  const [activeTab, setActiveTab] = useState('colorize'); // 'colorize' | 'history' | 'about'
  const { health, loading, refetchHealth } = useHealth(15000);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
      />

      <main className="main-content">
        <HealthBanner
          health={health}
          loading={loading}
          onRefresh={refetchHealth}
        />

        {activeTab === 'colorize' && (
          <ColorizePage
            health={health}
            onJobCompleted={() => {
              // Could trigger notifications or background history badge update
            }}
          />
        )}

        {activeTab === 'history' && <HistoryPage />}

        {activeTab === 'about' && <AboutPage />}
      </main>

      <footer 
        style={{ 
          borderTop: '1px solid var(--border-subtle)', 
          padding: '24px', 
          textAlign: 'center', 
          fontSize: '0.8rem', 
          color: 'var(--text-muted)',
          background: 'rgba(6, 9, 19, 0.8)'
        }}
      >
        <div>
          ISRO SIH1733 &bull; SAR-to-Optical Reconstruction for Comprehensive Insight using Deep Learning Model
        </div>
        <div style={{ marginTop: '4px', fontFamily: 'var(--font-mono)', fontSize: '0.74rem' }}>
          SSG-U-Net + ResNet18 Backbone &bull; Sobel Guidance &bull; MC-Dropout Uncertainty
        </div>
      </footer>
    </div>
  );
}

export default App;
