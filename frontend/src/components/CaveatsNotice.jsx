import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';

export function CaveatsNotice() {
  return (
    <div className="caveats-box">
      <Info size={22} className="caveats-box-icon" />
      <div className="caveats-text">
        <h5>Physical Representation & Uncertainty Disclaimers (ISRO SIH1733)</h5>
        <p style={{ marginBottom: '6px' }}>
          <strong>Learned Optical Synthesis:</strong> Predicted RGB values are synthesized representations conditioned on paired SAR-optical spatial geometries and structural Sobel edge guidance. They do not constitute recovered optical ground truth.
        </p>
        <p>
          <strong>Relative Confidence Proxy:</strong> The Monte Carlo Dropout uncertainty map reflects stochastic empirical variance across latent decoder feature activations. It represents a relative confidence proxy rather than a calibrated Bayesian posterior probability.
        </p>
      </div>
    </div>
  );
}
