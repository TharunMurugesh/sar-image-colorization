import numpy as np

def prepare_sar(arr: np.ndarray, target_channels: int = 3, use_lee: bool = False) -> np.ndarray:
    """
    Prepare SAR array (C, H, W).
    Adapts channel count to target_channels and normalizes using 2nd and 98th percentiles.
    Lee filtering is a placeholder if requested.
    """
    # Adapt channels
    c = arr.shape[0]
    if c == target_channels:
        out = arr.copy()
    elif c == 1:
        out = np.repeat(arr, target_channels, axis=0)
    elif c == 2 and target_channels == 3:
        out = np.stack([arr[0], arr[1], arr[0]], axis=0)
    elif c > target_channels:
        out = arr[:target_channels].copy()
    else:
        extra = target_channels - c
        out = np.concatenate([arr, np.repeat(arr[-1:], extra, axis=0)], axis=0)
    
    # Normalize per channel
    res = np.empty_like(out)
    for i in range(out.shape[0]):
        band = out[i]
        lo = np.percentile(band, 2)
        hi = np.percentile(band, 98)
        if hi - lo < 1e-9:
            res[i] = np.zeros_like(band)
        else:
            res[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
            
    # Optional Lee filtering could go here
    if use_lee:
        pass # Placeholder for Lee filtering
        
    return res
