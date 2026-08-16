import numpy as np
import random
import torchvision.transforms.functional as TF
import torch

def apply_transforms(sar: np.ndarray, target: np.ndarray = None, patch_size: tuple[int, int] = (256, 256), augment: bool = False):
    """
    Applies joint geometric transformations to SAR and target (if provided).
    Both inputs expected as numpy arrays (C, H, W).
    """
    # Crop or Pad
    ch, cw = patch_size
    _, h, w = sar.shape
    
    pad_h = max(0, ch - h)
    pad_w = max(0, cw - w)
    
    if pad_h > 0 or pad_w > 0:
        # np.pad expects ((before_1, after_1), (before_2, after_2), ...)
        pad_width = ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        sar = np.pad(sar, pad_width, mode='reflect')
        if target is not None:
            target = np.pad(target, pad_width, mode='reflect')
            
    _, h, w = sar.shape
    if h > ch or w > cw:
        # Center crop for inference, random crop if augmenting
        if augment:
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
        else:
            y = (h - ch) // 2
            x = (w - cw) // 2
            
        sar = sar[:, y:y+ch, x:x+cw]
        if target is not None:
            target = target[:, y:y+ch, x:x+cw]
            
    # Random augmentations
    if augment:
        # Convert to tensor for torchvision transforms
        sar_t = torch.from_numpy(sar.copy())
        if target is not None:
            tgt_t = torch.from_numpy(target.copy())
            
        if random.random() > 0.5:
            sar_t = TF.hflip(sar_t)
            if target is not None: tgt_t = TF.hflip(tgt_t)
            
        if random.random() > 0.5:
            sar_t = TF.vflip(sar_t)
            if target is not None: tgt_t = TF.vflip(tgt_t)
            
        k = random.randint(0, 3)
        if k > 0:
            sar_t = torch.rot90(sar_t, k, [1, 2])
            if target is not None: tgt_t = torch.rot90(tgt_t, k, [1, 2])
            
        sar = sar_t.numpy()
        if target is not None:
            target = tgt_t.numpy()
            
    return sar, target
