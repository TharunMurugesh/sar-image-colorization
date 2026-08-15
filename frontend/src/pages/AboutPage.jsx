import React from 'react';
import { Cpu, ShieldCheck, Zap, GitBranch, Layers, Sliders, Database } from 'lucide-react';

export function AboutPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      <div className="glass-card" style={{ padding: '28px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <div className="satellite-icon-ring">
            <Cpu size={22} />
          </div>
          <div>
            <div className="brand-tag">Deep Learning Architecture</div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              SSG-U-Net & Bayesian Uncertainty Architecture
            </h2>
          </div>
        </div>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '8px', lineHeight: 1.6 }}>
          The SAR-Structure-Guided U-Net (SSG-U-Net) resolves the ill-posed SAR-to-optical translation problem by enforcing high-frequency edge consistency and estimating pixel-wise epistemic uncertainty through Monte Carlo Dropout.
        </p>
      </div>

      <div className="image-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
        {/* Card 1: SSG-U-Net */}
        <div className="image-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--cyan-400)', marginBottom: '12px' }}>
            <Layers size={20} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              1. SAR Structural Guidance Module (SGM)
            </h3>
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
            A parallel structural branch computes Sobel gradient magnitudes across horizontal and vertical axes directly on the raw radar backscatter tensor. These high-frequency edge representations are injected into the decoder to preserve land-water boundaries, roads, and urban geometry.
          </p>
        </div>

        {/* Card 2: Joint Objective */}
        <div className="image-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--indigo-500)', marginBottom: '12px' }}>
            <Zap size={20} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              2. Composite Structural Loss
            </h3>
          </div>
          <div className="font-mono" style={{ background: 'rgba(0,0,0,0.4)', padding: '8px 12px', borderRadius: '6px', fontSize: '0.78rem', color: 'var(--cyan-400)', marginBottom: '10px' }}>
            L = 1.0·L1_RGB + 0.5·(1 - SSIM) + 0.1·L_struct
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
            Optimizes color fidelity with pixel-wise L1 loss while penalizing structural degradation via Multi-Scale Structural Similarity (SSIM) and edge distortion via Sobel-L1 loss.
          </p>
        </div>

        {/* Card 3: MC-Dropout */}
        <div className="image-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--amber-400)', marginBottom: '12px' }}>
            <ShieldCheck size={20} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              3. MC-Dropout Uncertainty
            </h3>
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
            During inference, spatial dropout layers (p=0.5) remain activated across M=10 stochastic forward passes. The empirical variance across sample predictions yields a per-pixel confidence proxy map without requiring expensive ensemble architectures.
          </p>
        </div>

        {/* Card 4: Trust Gated */}
        <div className="image-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--emerald-400)', marginBottom: '12px' }}>
            <Sliders size={20} />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              4. Trust-Gated Attenuation
            </h3>
          </div>
          <div className="font-mono" style={{ background: 'rgba(0,0,0,0.4)', padding: '8px 12px', borderRadius: '6px', fontSize: '0.78rem', color: 'var(--emerald-400)', marginBottom: '10px' }}>
            &alpha; = exp(-&sigma;&sup2; / &tau;) &bull; &Icirc; = &alpha;&middot;RGB + (1-&alpha;)&middot;SAR_gray
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
            Where the model exhibits high epistemic variance (such as rare terrain classes or high speckle noise), color saturation is smoothly attenuated back toward physical radar grayscale.
          </p>
        </div>
      </div>
    </div>
  );
}
