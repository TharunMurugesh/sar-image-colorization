import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SobelFilterLoss(nn.Module):
    """Computes L1 loss on Sobel edges between prediction and target."""
    def __init__(self, in_channels=3):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.in_channels = in_channels
        self.l1 = nn.L1Loss()

    def get_edges(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return self.l1(pred_edges, target_edges)

class SARColorizationLoss(nn.Module):
    """
    Joint color + structure training objective:
    L = 1.0 * L1_RGB + 0.5 * (1 - SSIM) + 0.1 * L_structure
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        # Ensure data range is specified for SSIM
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.structure_loss = SobelFilterLoss(in_channels=3)
        
    def forward(self, pred, target):
        # 1. L1 RGB
        l1 = self.l1_loss(pred, target)
        
        # 2. SSIM
        ssim_val = self.ssim_metric(pred, target)
        ssim_loss = 1.0 - ssim_val
        
        # 3. Structure Loss
        struct_loss = self.structure_loss(pred, target)
        
        # Combine
        total_loss = 1.0 * l1 + 0.5 * ssim_loss + 0.1 * struct_loss
        
        return total_loss, {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "struct": struct_loss.item()
        }
