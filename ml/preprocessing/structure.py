import torch
import torch.nn.functional as F

def build_sar_structure_map(sar_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes normalized structural guidance map from normalized SAR tensor.
    sar_tensor: (B, C, H, W)
    Returns: structure map (B, 1, H, W) in [0, 1] range.
    """
    device = sar_tensor.device
    
    # 1. Average the available bands
    avg_sar = sar_tensor.mean(dim=1, keepdim=True) # (B, 1, H, W)
    
    # 2. Fixed Sobel gradient magnitude
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    
    grad_x = F.conv2d(avg_sar, kernel_x, padding=1)
    grad_y = F.conv2d(avg_sar, kernel_y, padding=1)
    
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
    # 3. Normalize gradient map to [0,1] per item in batch
    B = magnitude.shape[0]
    out = torch.empty_like(magnitude)
    for i in range(B):
        m = magnitude[i]
        m_min = m.min()
        m_max = m.max()
        if m_max - m_min < 1e-9:
            out[i] = torch.zeros_like(m)
        else:
            out[i] = (m - m_min) / (m_max - m_min)
            
    return out
