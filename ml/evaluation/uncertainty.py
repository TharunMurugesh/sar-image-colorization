import torch
import torch.nn.functional as F

def mc_dropout_inference(model, x, num_samples=10):
    """
    Performs Monte Carlo Dropout inference to estimate prediction and uncertainty.
    
    Args:
        model: The SSGUNet model
        x: Input tensor (B, C, H, W)
        num_samples: Number of forward passes
        
    Returns:
        mean_pred: Mean prediction (B, C, H, W)
        uncertainty: Variance across samples (B, 1, H, W)
        edges: The structural edges from the SAR input
    """
    # Ensure dropout layers are active
    model.train()
    
    preds = []
    first_edges = None
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred, edges = model(x)
            preds.append(pred)
            if first_edges is None:
                first_edges = edges
                
    preds = torch.stack(preds, dim=0) # (num_samples, B, C, H, W)
    mean_pred = preds.mean(dim=0)
    variance = preds.var(dim=0)
    
    # Use mean variance across channels as a scalar uncertainty map
    uncertainty = variance.mean(dim=1, keepdim=True)
    
    return mean_pred, uncertainty, first_edges

def trust_gated_rendering(pred_rgb, sar_input, uncertainty, tau=0.05):
    """
    Attenuates low-confidence regions toward SAR grayscale.
    
    Args:
        pred_rgb: Predicted color image (B, 3, H, W)
        sar_input: Original SAR input (B, 3, H, W)
        uncertainty: Estimated uncertainty map (B, 1, H, W)
        tau: Temperature parameter controlling attenuation strength.
        
    Returns:
        gated_pred: The blended image
    """
    # Convert SAR input to grayscale representation
    sar_gray = sar_input.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    
    # Compute trust mask (alpha)
    # Low uncertainty -> alpha close to 1 (trust pred)
    # High uncertainty -> alpha close to 0 (trust SAR)
    alpha = torch.exp(-uncertainty / tau)
    
    gated_pred = alpha * pred_rgb + (1 - alpha) * sar_gray
    return gated_pred
