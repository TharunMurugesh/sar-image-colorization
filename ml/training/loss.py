import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


def _rgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] -> linear RGB via IEC 61966-2-1 transfer function."""
    mask = rgb <= 0.04045
    return torch.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).clamp(0.0, 1.0)


def _linear_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """Linear sRGB -> CIE XYZ (D65, BT.709). Input/output: (B, 3, H, W)."""
    M = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=rgb.dtype, device=rgb.device,
    )
    return (rgb.permute(0, 2, 3, 1) @ M.T).permute(0, 3, 1, 2)


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    """CIE XYZ -> CIE Lab (D65 white point)."""
    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device).view(1, 3, 1, 1)
    t     = xyz / white
    delta = 6.0 / 29.0
    f = torch.where(
        t > (delta ** 3),
        t.clamp(min=1e-9).pow(1.0 / 3.0),
        t / (3 * delta ** 2) + (4.0 / 29.0),
    )
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] -> CIE Lab. Fully differentiable. Input: (B,3,H,W)."""
    return _xyz_to_lab(_linear_to_xyz(_rgb_to_linear(rgb)))


class SobelFilterLoss(nn.Module):
    """L1 loss on Sobel edge magnitudes for structural boundary consistency."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("weight_x", kx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer("weight_y", ky.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.l1 = nn.L1Loss()

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(self._edges(pred), self._edges(target))


class LabColorLoss(nn.Module):
    """L1 loss on CIE-Lab a*b* chroma channels.

    Targeting chroma directly forces the model to learn correct hue
    (vegetation green, water blue) rather than converging to neutral grey.
    """

    def __init__(self, weight_ab: float = 1.0):
        super().__init__()
        self.weight_ab = weight_ab
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = rgb_to_lab(pred.float().clamp(0.0, 1.0))
        tgt_lab  = rgb_to_lab(target.float().clamp(0.0, 1.0))
        # Normalise a*, b* from [-128, 127] to ~[-1, 1]
        ab_pred = pred_lab[:, 1:, :, :] / 128.0
        ab_tgt  = tgt_lab[:, 1:, :, :] / 128.0
        return self.l1(ab_pred, ab_tgt) * self.weight_ab


class SARColorizationLoss(nn.Module):
    """Composite SAR-to-Optical reconstruction loss.

    L = λ_l1 * L1_RGB + λ_ssim * (1-SSIM) + λ_lab * L_lab + λ_struct * L_struct
    """

    def __init__(
        self,
        lambda_l1: float    = 0.70,
        lambda_ssim: float  = 0.30,
        lambda_lab: float   = 0.15,
        lambda_struct: float = 0.05,
    ):
        super().__init__()
        self.lambda_l1     = lambda_l1
        self.lambda_ssim   = lambda_ssim
        self.lambda_lab    = lambda_lab
        self.lambda_struct = lambda_struct

        self.l1_loss       = nn.L1Loss()
        self.ssim_metric   = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lab_loss      = LabColorLoss()
        self.structure_loss = SobelFilterLoss(in_channels=3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """Returns (total_loss, dict) where dict has keys: l1, ssim, lab, struct, total."""
        l1       = self.l1_loss(pred, target)
        ssim_val = self.ssim_metric(pred.float(), target.float())
        lab      = self.lab_loss(pred, target)
        struct   = self.structure_loss(pred, target)

        total = (
            self.lambda_l1     * l1
            + self.lambda_ssim * (1.0 - ssim_val)
            + self.lambda_lab  * lab
            + self.lambda_struct * struct
        )

        return total, {
            "l1": l1.item(), "ssim": ssim_val.item(),
            "lab": lab.item(), "struct": struct.item(), "total": total.item(),
        }
