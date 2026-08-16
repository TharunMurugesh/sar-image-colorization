import numpy as np

def prepare_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Prepare RGB target. Ensures values are in range [0, 1].
    Expected shape: (3, H, W)
    """
    if arr.max() > 1.0:
        return arr / 255.0
    return arr
