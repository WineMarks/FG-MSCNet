import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .modules import InputStem, EncoderStage, DecoderStage, DDG_Bridge


class FG_MSCNet(nn.Module):
    def __init__(self, num_classes=1, img_size=384):
        super(FG_MSCNet, self).__init__()

        self.img_size = img_size

        self.stem = InputStem(in_channels=3, srm_channels=30, base_channels=64)

        self.enc1 = EncoderStage(in_channels=64, out_channels=64,
                                 sra_ratios=(8, 16))
        self.enc2 = EncoderStage(in_channels=64, out_channels=128,
                                 sra_ratios=(4, 8))
        self.enc3 = EncoderStage(in_channels=128, out_channels=256,
                                 sra_ratios=(2, 4))
        self.enc4 = EncoderStage(in_channels=256, out_channels=512,
                                 sra_ratios=(1, 2))
        self.bridge = DDG_Bridge(dim=512, resolution=(img_size // 16, img_size // 16))

        self.dec1 = DecoderStage(in_channels=512, skip_channels=256, out_channels=256)

        self.dec2 = DecoderStage(in_channels=256, skip_channels=128, out_channels=128)

        self.dec3 = DecoderStage(in_channels=128, skip_channels=64, out_channels=64)

        self.dec4 = DecoderStage(in_channels=64, skip_channels=64, out_channels=32)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_bias_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        nn.init.constant_(self.adaptive_bias_head[-1].weight, 0)
        nn.init.constant_(self.adaptive_bias_head[-1].bias, 0)

        self.final_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def check_input_size(self, x):
        H, W = x.shape[2], x.shape[3]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, H, W

    def forward(self, x, return_features=False):
        x_padded, original_H, original_W = self.check_input_size(x)
        stem_feat = checkpoint(self.stem, x_padded, use_reentrant=False)
        enc1_feat = checkpoint(self.enc1, stem_feat, use_reentrant=False)
        enc2_feat = checkpoint(self.enc2, enc1_feat, use_reentrant=False)
        enc3_feat = checkpoint(self.enc3, enc2_feat, use_reentrant=False)
        enc4_feat = checkpoint(self.enc4, enc3_feat, use_reentrant=False)

        bridge_feat = checkpoint(self.bridge, enc4_feat, use_reentrant=False)
        global_feat = self.global_pool(bridge_feat).flatten(1)
        adaptive_bias = self.adaptive_bias_head(global_feat).view(-1, 1, 1, 1)

        masks = []
        dec1_out = checkpoint(self.dec1, bridge_feat, enc3_feat, None, use_reentrant=False)
        masks.append(dec1_out['mask'])
        
        dec2_out = checkpoint(self.dec2, dec1_out['out'], enc2_feat, dec1_out['mask'], use_reentrant=False)
        masks.append(dec2_out['mask'])
        
        dec3_out = checkpoint(self.dec3, dec2_out['out'], enc1_feat, dec2_out['mask'], use_reentrant=False)
        masks.append(dec3_out['mask'])
        
        dec4_out = checkpoint(self.dec4, dec3_out['out'], stem_feat, dec3_out['mask'], use_reentrant=False)
        final_mask = self.final_head(dec4_out['out'])  # [B, 1, H, W]
        final_mask = final_mask + adaptive_bias
        masks = [m + adaptive_bias for m in masks]
        if final_mask.shape[2] != original_H or final_mask.shape[3] != original_W:
            final_mask = final_mask[:, :, :original_H, :original_W]
            for i in range(len(masks)):
                scale = 2 ** (3 - i) 
                masks[i] = masks[i][:, :, :original_H // scale, :original_W // scale]
        if return_features:
            return {
                'mask_final': final_mask,
                'masks_aux': masks,
                'stem_noise': stem_feat,
                'freq_gate': self.bridge.gate_conv[1].output if hasattr(self.bridge, 'gate_conv') else None,
                'dec_subbands': dec4_out['subbands'],
                'adaptive_bias': adaptive_bias
            }
        else:
            return [final_mask, *reversed(masks)]
            # [Final(H), Mask3(H/2), Mask2(H/4), Mask1(H/8)]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FG_MSCNet().to(device)
    dummy_input = torch.randn(2, 3, 384, 384).to(device)

    print("Testing Forward Pass...")
    outputs = model(dummy_input)

    print(f"Output count: {len(outputs)} (Should be 4 for deep supervision)")
    print(f"Final Mask Shape: {outputs[0].shape}")
    print(f"Coarsest Mask Shape: {outputs[-1].shape}")

    # Check Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")