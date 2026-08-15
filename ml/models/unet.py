import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SobelFilter(nn.Module):
    """Extracts structural edges using Sobel operators."""
    def __init__(self, in_channels=3):
        super().__init__()
        # Sobel kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Expand for in_channels (depthwise convolution)
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.in_channels = in_channels

    def forward(self, x):
        # x shape: (B, C, H, W)
        grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_p=0.5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p

    def forward(self, x, skip):
        x = self.up(x)
        # Handle cases where input dimensions aren't perfect multiples of 2
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([skip, x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.use_dropout:
            # Standard nn.Dropout2d is handled by calling model.train() during MC inference.
            x = F.dropout2d(x, p=self.dropout_p, training=self.training)
            
        return x

class SARStructuralGuidanceModule(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.sobel = SobelFilter(in_channels=in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        edges = self.sobel(x)
        guidance = self.conv(edges) # (B, features, H/4, W/4)
        return edges, guidance

class SSGUNet(nn.Module):
    """
    SAR-Structure-Guided U-Net with ResNet-18 Encoder and MC-Dropout.
    """
    def __init__(self, in_channels=3, out_channels=3, mc_dropout=True):
        super().__init__()
        
        self.mc_dropout = mc_dropout
        
        # --- Encoder (ResNet-18) ---
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first conv if in_channels != 3
        if in_channels != 3:
            self.enc0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.enc0.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        else:
            self.enc0 = resnet.conv1
            
        self.enc_bn0 = resnet.bn1
        self.enc_relu = resnet.relu
        self.enc_pool = resnet.maxpool
        
        self.enc1 = resnet.layer1 # 64
        self.enc2 = resnet.layer2 # 128
        self.enc3 = resnet.layer3 # 256
        self.enc4 = resnet.layer4 # 512
        
        # --- Structural Guidance Module ---
        self.sgm = SARStructuralGuidanceModule(in_channels=in_channels, features=64)
        
        # --- Bottleneck ---
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # --- Decoder ---
        # Decoder 4: up(512) + skip3(256) -> 256
        self.dec4 = DecoderBlock(512, 256, use_dropout=mc_dropout, dropout_p=0.5)
        # Decoder 3: up(256) + skip2(128) -> 128
        self.dec3 = DecoderBlock(256, 128, use_dropout=mc_dropout, dropout_p=0.5)
        # Decoder 2: up(128) + skip1(64) + SGM(64) -> 64
        self.dec2 = DecoderBlock(128, 64, use_dropout=False) 
        # Decoder 1: up(64) + skip0(64) -> 64
        self.dec1 = DecoderBlock(64, 64, use_dropout=False)
        
        # We need an extra upsampling to get back to original resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        
        # SGM
        edges, guidance = self.sgm(x)
        
        # Encoder
        x0 = self.enc_relu(self.enc_bn0(self.enc0(x))) # 64, H/2, W/2
        xp = self.enc_pool(x0) # 64, H/4, W/4
        
        x1 = self.enc1(xp) # 64, H/4, W/4
        x2 = self.enc2(x1) # 128, H/8, W/8
        x3 = self.enc3(x2) # 256, H/16, W/16
        x4 = self.enc4(x3) # 512, H/32, W/32
        
        # Bottleneck
        b = self.bottleneck_conv(x4)
        if self.mc_dropout:
            b = F.dropout2d(b, p=0.5, training=self.training)
            
        # Decoder
        d4 = self.dec4(b, x3) # 256, H/16, W/16
        d3 = self.dec3(d4, x2) # 128, H/8, W/8
        
        # Inject structural guidance here! Guidance is (64, H/4, W/4) which matches x1 size.
        x1_guided = x1 + guidance
        d2 = self.dec2(d3, x1_guided) # 64, H/4, W/4
        
        d1 = self.dec1(d2, x0) # 64, H/2, W/2
        
        out = self.final_up(d1) # 32, H, W
        out = self.final_conv(out) # 3, H, W
        
        # Output values bounded [0, 1] using Sigmoid
        out = torch.sigmoid(out)
        
        return out, edges
