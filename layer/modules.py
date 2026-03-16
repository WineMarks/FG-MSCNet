import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from .srm import SRMConv2d_30
from .wavelet import DWT_Downsampling, IDWT_Upsampling
from .dct import DCTLayer

class InputStem(nn.Module):     # v2
    def __init__(self, in_channels=3, srm_channels=30, base_channels=64):
        super().__init__()
        
        self.srm_layer = SRMConv2d_30(in_channels)
        mid_channels = base_channels // 2  # 32
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )
        self.srm_proj = nn.Sequential(
            nn.Conv2d(srm_channels, mid_channels, kernel_size=1, bias=False), # 1x1 足以整合通道信息
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(mid_channels * 2, base_channels, kernel_size=1, bias=False)
        self.fusion_block = ResBlock(base_channels)

    def forward(self, x):
        noise = self.srm_layer(x)       # [B, 30, H, W]
        
        feat_rgb = self.rgb_proj(x)     # [B, 32, H, W]
        feat_srm = self.srm_proj(noise) # [B, 32, H, W]
        cat_for_gate = torch.cat([feat_rgb, feat_srm], dim=1) # [B, 64, H, W]
        gate_map = self.gate_conv(cat_for_gate) # [B, 1, H, W] (0~1)
        feat_srm_gated = feat_srm * (1 + gate_map) 
        feat_fused = torch.cat([feat_rgb, feat_srm_gated], dim=1) # [B, 64, H, W]
        out = self.fusion_conv(feat_fused) # [B, 64, H, W]
        out = self.fusion_block(out)       # [B, 64, H, W] ResBlock
        
        return out


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + res

class FrequencyInjector(nn.Module):     # v2
    def __init__(self, channels, reduction=4):
        super(FrequencyInjector, self).__init__()
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        with autocast(enabled=False):
            x_fp32 = x.float()
            freq_spec = torch.fft.rfft2(x_fp32, norm='ortho')
            freq_mag = torch.abs(freq_spec)
        freq_mag = freq_mag.to(x.dtype)
        gate = self.freq_gate(freq_mag)
        with autocast(enabled=False):
            freq_spec_gated = freq_spec * gate.float()
            x_gated = torch.fft.irfft2(freq_spec_gated, s=(x.shape[2], x.shape[3]), norm='ortho')
        return x + x_gated.to(x.dtype)

class FG_SRA(nn.Module):        # v2
    def __init__(self, dim, num_heads, sr_ratios=(8, 16)):
        super(FG_SRA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.d = dim // num_heads
        self.scale = self.d ** -0.5
        self.freq_inject = FrequencyInjector(dim)

        self.q = nn.Linear(dim, dim, bias=True)

        self.sr1_ratio = sr_ratios[0]
        self.sr1_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratios[0], stride=sr_ratios[0])
        self.sr1_norm = nn.LayerNorm(dim)
        self.kv1 = nn.Linear(dim, dim * 2, bias=True)

        self.sr2_ratio = sr_ratios[1]
        self.sr2_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratios[1], stride=sr_ratios[1])
        self.sr2_norm = nn.LayerNorm(dim)
        self.kv2 = nn.Linear(dim, dim * 2, bias=True)

        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.last_attn_map = None

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x_freq = self.freq_inject(x) 

        x_flat = x_freq.flatten(2).transpose(1, 2)   
        q = self.q(x_flat).reshape(B, N, self.num_heads, self.d).permute(0, 2, 1, 3) 

        x_fine = self.sr1_conv(x_freq)
        shape_fine = x_fine.shape
        x_fine_flat = self.sr1_norm(x_fine.flatten(2).transpose(1, 2))
        k1, v1 = self.kv1(x_fine_flat).chunk(2, dim=-1)

        x_coarse = self.sr2_conv(x_freq)
        shape_coarse = x_coarse.shape
        x_coarse_flat = self.sr2_norm(x_coarse.flatten(2).transpose(1, 2))
        k2, v2 = self.kv2(x_coarse_flat).chunk(2, dim=-1)

        v1_spatial = v1.transpose(1, 2).reshape(B, C, shape_fine[2], shape_fine[3]).contiguous()
        v1 = self.v_dwconv(v1_spatial).flatten(2).transpose(1, 2)
        
        v2_spatial = v2.transpose(1, 2).reshape(B, C, shape_coarse[2], shape_coarse[3]).contiguous()
        v2 = self.v_dwconv(v2_spatial).flatten(2).transpose(1, 2)

        k_cat = torch.cat([k1, k2], dim=1)
        v_cat = torch.cat([v1, v2], dim=1)

        k = k_cat.reshape(B, -1, self.num_heads, self.d).permute(0, 2, 1, 3)
        v = v_cat.reshape(B, -1, self.num_heads, self.d).permute(0, 2, 1, 3)

        x_out = F.scaled_dot_product_attention(q, k, v)

        if not self.training:
            with torch.no_grad():
                q_vis, k_vis = q[0:1, 0:1], k[0:1, 0:1]
                attn_map = (q_vis @ k_vis.transpose(-2, -1)) * self.scale
                self.last_attn_map = attn_map.softmax(dim=-1).detach().cpu()

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)

        return x_out.transpose(1, 2).reshape(B, C, H, W) + x

class EncoderStage(nn.Module):      # v2
    def __init__(self, in_channels, out_channels, num_heads=8, sra_ratios=(8, 16)):
        super().__init__()
        self.dwt = DWT_Downsampling()

        self.low_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.high_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.res_blocks = nn.Sequential(
            ResBlock(out_channels),
            ResBlock(out_channels)
        )

        self.attn = FG_SRA(
            dim=out_channels,
            num_heads=num_heads,
            sr_ratios=sra_ratios
        )

    def forward(self, x):
        C = x.shape[1]
        dwt_out = self.dwt(x) 
        x_ll = dwt_out[:, 0:C, :, :]         # [B, C, H/2, W/2]
        x_high = dwt_out[:, C:, :, :]        # [B, 3C, H/2, W/2]
        feat_ll = self.low_proj(x_ll)        # [B, C_out, H/2, W/2]
        feat_high = self.high_proj(x_high)   # [B, C_out, H/2, W/2]
        x_fused = torch.cat([feat_ll, feat_high], dim=1) # [B, 2*C_out, H/2, W/2]
        x = self.fuse_proj(x_fused)          # [B, C_out, H/2, W/2]
        x = self.res_blocks(x)
        x = self.attn(x) 

        return x

class DDG_Bridge(nn.Module):
    def __init__(self, dim, resolution):
        super().__init__()
        self.dim = dim
        self.H, self.W = resolution

        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

        self.norm_freq = nn.LayerNorm(dim)

        self.freq_channel_mix = nn.Conv2d(dim, dim, 1)

        self.freq_H = self.H
        self.freq_W = self.W // 2 + 1

        self.complex_weight = nn.Parameter(
            torch.randn(dim, self.freq_H, self.freq_W, 2, dtype=torch.float32) * 0.02
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.final_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W, f"Resolution mismatch: expected {self.H}x{self.W}, got {H}x{W}"
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm_spatial(x_flat)

        spatial_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
        spatial_out = spatial_out.transpose(1, 2).view(B, C, H, W)
        spatial_out = x + spatial_out

        x_mixed = self.freq_channel_mix(x)

        with autocast(enabled=False):
            x_mixed_fp32 = x_mixed.float()
            x_fft = torch.fft.rfft2(x_mixed_fp32, norm='ortho')
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            x_weighted = x_fft * weight
            freq_out_fp32 = torch.fft.irfft2(x_weighted, s=(H, W), norm='ortho')

        freq_out = freq_out_fp32.to(x.dtype)
        freq_out = x + freq_out

        gate = self.gate_conv(freq_out) 
        out = spatial_out * gate + freq_out
        out = self.final_proj(out)

        return out


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.subband_pred = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self.pred_bn = nn.BatchNorm2d(out_channels * 4)
        self.idwt = IDWT_Upsampling()
        self.mask_attn_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.res_blocks = nn.Sequential(
            ResBlock(out_channels),
            ResBlock(out_channels)
        )
        self.mask_head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, skip_feat, prev_mask=None):
        x_pred = self.pred_bn(self.subband_pred(x)).contiguous()  # [B, out_C*4, H, W]
        x_up, subbands = self.idwt(x_pred)
        if prev_mask is not None:
            mask_up = F.interpolate(prev_mask, scale_factor=2, mode='bilinear', align_corners=False)
            attn_map = self.mask_attn_conv(mask_up)
            x_up = x_up * (1 + self.alpha * attn_map)
        cat_feat = torch.cat([x_up, skip_feat], dim=1)
        x_fused = self.fuse_conv(cat_feat)
        x_out = self.res_blocks(x_fused)
        curr_mask = self.mask_head(x_out)
        return {
            'out': x_out,
            'mask': curr_mask,
            'subbands': subbands
        }