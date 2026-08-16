import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class SSGUNet(smp.Unet):
    """
    SAR-Structure-Guided U-Net.
    Subclasses smp.Unet with resnet18 encoder.
    Accepts (x, sar_structure_map) and injects structure_map into the first decoder fusion block.
    Supports MC-Dropout natively.
    """
    def __init__(self, in_channels=3, out_channels=3, mc_dropout=True):
        super().__init__(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
        )
        self.mc_dropout = mc_dropout
        
        # In smp.Unet with resnet18:
        # features list shapes from encoder (example for 256x256 input):
        # features[0]: (B, in_channels, 256, 256) # Identity
        # features[1]: (B, 64, 128, 128)
        # features[2]: (B, 64, 64, 64)
        # features[3]: (B, 128, 32, 32)
        # features[4]: (B, 256, 16, 16)
        # features[5]: (B, 512, 8, 8)
        
        # The first decoder fusion uses features[-1] (512) and features[-2] (256, 16x16)
        # We want to concatenate our structure map to features[-2].
        # features[-2] has 256 channels. We add 1 channel (structure map) = 257.
        # We need a 1x1 conv to bring it back to 256 channels.
        
        f_minus_2_channels = self.encoder.out_channels[-2]
        self.sgm_fusion = nn.Conv2d(f_minus_2_channels + 1, f_minus_2_channels, kernel_size=1)
        
        # Change decoder dropout if requested
        if mc_dropout:
            # We add a Dropout layer to each decoder block
            for block in self.decoder.blocks:
                # Add dropout after the second conv block inside the decoder block
                block.conv2.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, x, structure_map=None):
        """
        x: (B, C, H, W) SAR tensor
        structure_map: (B, 1, H, W) normalized SAR structure map
        """
        features = self.encoder(x)
        
        if structure_map is not None:
            # Resize structure map to features[-2] size
            f2_size = features[-2].shape[2:]
            resized_structure = F.interpolate(structure_map, size=f2_size, mode='bilinear', align_corners=False)
            
            # Concatenate and fuse
            f2_guided = torch.cat([features[-2], resized_structure], dim=1)
            features[-2] = F.relu(self.sgm_fusion(f2_guided))
            
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        
        # Bound output to [0, 1] using sigmoid
        return torch.sigmoid(masks)
