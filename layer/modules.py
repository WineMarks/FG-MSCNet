import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from .srm import SRMConv2d_30
from .wavelet import DWT_Downsampling, IDWT_Upsampling
from .dct import DCTLayer


# class InputStem(nn.Module):     # v1
#     def __init__(self, in_channels=3, srm_channels=30, base_channels=64):
#         super().__init__()
#         # 1. 实例化我们之前写的 SRM 模块
#         self.srm_layer = SRMConv2d_30(in_channels)

#         # 2. 融合卷积
#         # 输入通道 = 3 (RGB) + 30 (SRM) = 33
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(in_channels + srm_channels, base_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(base_channels),
#             nn.GELU()  # GELU 比 ReLU 在 Transformer/CNN 混搭结构中表现更好
#         )

#     def forward(self, x):
#         # x: [B, 3, H, W]

#         # 1. 提取噪声 (注意：SRM内部已经处理了 *255 和 BN)
#         noise = self.srm_layer(x)  # [B, 30, H, W]

#         # 2. 拼接 (Concatenation)
#         # 在通道维度 (dim=1) 拼接
#         cat_feat = torch.cat([x, noise], dim=1)  # [B, 33, H, W]

#         # 3. 融合映射
#         out = self.fuse_conv(cat_feat)  # [B, 64, H, W]

#         return out
class InputStem(nn.Module):     # v2
    def __init__(self, in_channels=3, srm_channels=30, base_channels=64):
        super().__init__()
        
        # 1. 固定的噪声提取器
        self.srm_layer = SRMConv2d_30(in_channels)
        # 2. 双流投影 (Dual-stream Projection)
        # 显存充足，我们先将两者映射到统一的特征维度，而不是直接拼接
        mid_channels = base_channels // 2  # 32
        # RGB 流: 关注结构、颜色、语义
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )
        # SRM 流: 关注高频噪声统计
        self.srm_proj = nn.Sequential(
            nn.Conv2d(srm_channels, mid_channels, kernel_size=1, bias=False), # 1x1 足以整合通道信息
            nn.BatchNorm2d(mid_channels),
            nn.GELU()
        )
        # 3. 空间注意力门控 (Spatial Attention Gate)
        # 利用 RGB 的结构信息来“清洗” SRM 特征（抑制文字边缘的误报）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 4. 深度融合 (Deepened Fusion)
        # 将简单的 Conv3x3 替换为 ResBlock，感受野从 3x3 扩大到 5x5
        self.fusion_conv = nn.Conv2d(mid_channels * 2, base_channels, kernel_size=1, bias=False)
        self.fusion_block = ResBlock(base_channels)

    def forward(self, x):
        # x: [B, 3, H, W]
        
        # --- Step 1: 特征解耦 ---
        noise = self.srm_layer(x)       # [B, 30, H, W]
        
        feat_rgb = self.rgb_proj(x)     # [B, 32, H, W]
        feat_srm = self.srm_proj(noise) # [B, 32, H, W]
        
        # --- Step 2: 交叉门控 (Cross-Gating) ---
        # 我们认为 RGB 包含了“什么是文字”的语义，可以用来指导 SRM
        cat_for_gate = torch.cat([feat_rgb, feat_srm], dim=1) # [B, 64, H, W]
        gate_map = self.gate_conv(cat_for_gate) # [B, 1, H, W] (0~1)
        
        # 门控机制：不仅是抑制，也允许网络学习“增强”
        # feat_srm = feat_srm * gate_map + feat_srm (残差式门控)
        feat_srm_gated = feat_srm * (1 + gate_map) 
        
        # --- Step 3: 深度融合 ---
        # 再次拼接
        feat_fused = torch.cat([feat_rgb, feat_srm_gated], dim=1) # [B, 64, H, W]
        
        # 融合并加深
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
        return x + res  # 残差连接


# class FrequencyInjector(nn.Module):     # v1
#     """
#     频域注入模块：分析特征图的频域成分，生成通道注意力权重。
#     类似于 SE-Net，但挤压 (Squeeze) 的过程是在频域完成的。
#     """

#     def __init__(self, channels, spatial_size, reduction=4):
#         super(FrequencyInjector, self).__init__()

#         # 1. DCT 变换层
#         self.dct = DCTLayer(spatial_size, spatial_size)

#         # 2. 频域感知 MLP (Frequency-Aware MLP)
#         # 我们不使用全图频谱，而是学习频域的统计特征（如方差/能量）
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         """
#         Input: [B, C, H_small, W_small] (缩减后的特征)
#         Output: [B, C, H_small, W_small] (注入频域信息后的特征)
#         """
#         B, C, H, W = x.shape

#         # Step 1: 转到频域
#         freq_spec = self.dct(x)  # [B, C, H, W]

#         # Step 2: 提取频域指纹 (Global Spectral Fingerprint)
#         # 简单的 AvgPool 会丢失频率分布，这里我们计算“频带能量”
#         # 对 H, W 维度求和，得到每个通道的总频域能量
#         freq_energy = freq_spec.abs().sum(dim=[2, 3])  # [B, C]

#         # Step 3: 生成门控权重
#         gate = self.fc(freq_energy).view(B, C, 1, 1)  # [B, C, 1, 1]

#         # Step 4: 注入 (Residual Injection)
#         # 让原始空间特征 加上 被频域加权过的特征
#         return x + (x * gate)
class FrequencyInjector(nn.Module):     # v2
    """
    【升级 1】可学习的动态频域滤波器 (Learnable Spectral Filter)
    直接在频域空间应用门控，再通过 IFFT 还原，不损失任何高频位置信息。
    """
    def __init__(self, channels, reduction=4):
        super(FrequencyInjector, self).__init__()
        # 频域门控网络：学习通道间的频率相关性
        self.freq_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input: [B, C, H, W] (高清全分辨率特征)
        """
        # Step 1: 强制 FP32 进行 FFT，防止 ComplexHalf 报错
        with autocast(enabled=False):
            x_fp32 = x.float()
            # 转至频域 [B, C, H, W/2+1]
            freq_spec = torch.fft.rfft2(x_fp32, norm='ortho')
            # 获取频域的振幅 (Magnitude)
            freq_mag = torch.abs(freq_spec)
            
        # Step 2: 生成频域门控掩码 (恢复至 x 的 dtype 进行卷积计算)
        freq_mag = freq_mag.to(x.dtype)
        gate = self.freq_gate(freq_mag) # [B, C, H, W/2+1]

        # Step 3: 在频域直接应用门控 (滤波)
        with autocast(enabled=False):
            # 将门控应用到原始的复数频谱上
            freq_spec_gated = freq_spec * gate.float()
            # IFFT 逆变换回空间域 [B, C, H, W]
            x_gated = torch.fft.irfft2(freq_spec_gated, s=(x.shape[2], x.shape[3]), norm='ortho')

        # Step 4: 残差注入
        return x + x_gated.to(x.dtype)


# class FG_SRA(nn.Module):        # v1
#     """
#     Frequency-Guided Spatial Reduction Attention (FG-SRA)
#     支持双流多尺度 K-V 生成与拼接。
#     """

#     def __init__(self, dim, num_heads, input_resolution, sr_ratios=(8, 16)):
#         super(FG_SRA, self).__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         self.d = dim // num_heads
#         self.scale = self.d ** -0.5
#         self.input_resolution = input_resolution  # (H, W) tuple

#         # Query: 保持全分辨率
#         self.q = nn.Linear(dim, dim, bias=True)

#         # --- Scale 1: Fine Grain (e.g., R=8) ---
#         self.sr1_ratio = sr_ratios[0]
#         # 使用 Stride Conv 进行降采样，比 AvgPool 更能保留篡改边缘
#         self.sr1_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratios[0], stride=sr_ratios[0])
#         self.sr1_norm = nn.LayerNorm(dim)
#         # 频域注入器 1
#         self.freq_inject1 = FrequencyInjector(dim, input_resolution[0] // sr_ratios[0])
#         self.kv1 = nn.Linear(dim, dim * 2, bias=True)  # 生成 K1, V1

#         # --- Scale 2: Coarse Grain (e.g., R=16) ---
#         self.sr2_ratio = sr_ratios[1]
#         self.sr2_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratios[1], stride=sr_ratios[1])
#         self.sr2_norm = nn.LayerNorm(dim)
#         # 频域注入器 2
#         self.freq_inject2 = FrequencyInjector(dim, input_resolution[0] // sr_ratios[1])
#         self.kv2 = nn.Linear(dim, dim * 2, bias=True)  # 生成 K2, V2

#         # Final Projection
#         self.proj = nn.Linear(dim, dim)

#         # Debug / Visualization Buffer
#         self.last_attn_map = None

#     def forward(self, x):
#         """
#         x: [B, C, H, W]  (输入特征图)
#         """
#         B, C, H, W = x.shape
#         N = H * W

#         # 1. 准备 Query (Flatten & Linear)
#         # [B, C, H, W] -> [B, N, C]
#         x_flat = x.flatten(2).transpose(1, 2)
#         q = self.q(x_flat).reshape(B, N, self.num_heads, self.d).permute(0, 2, 1, 3)  # [B, Heads, N, d]

#         # --- Branch 1: Fine-Grained K/V ---
#         # Spatial Reduction: [B, C, H, W] -> [B, C, H/8, W/8]
#         x_fine = self.sr1_conv(x)
#         # Frequency Injection
#         x_fine = self.freq_inject1(x_fine)
#         # Flatten & Norm
#         x_fine_flat = x_fine.flatten(2).transpose(1, 2)  # [B, N_fine, C]
#         x_fine_flat = self.sr1_norm(x_fine_flat)
#         # Generate K1, V1
#         k1, v1 = self.kv1(x_fine_flat).chunk(2, dim=-1)  # Each: [B, N_fine, C]

#         # --- Branch 2: Coarse-Grained K/V ---
#         # Spatial Reduction: [B, C, H, W] -> [B, C, H/16, W/16]
#         x_coarse = self.sr2_conv(x)
#         # Frequency Injection
#         x_coarse = self.freq_inject2(x_coarse)
#         # Flatten & Norm
#         x_coarse_flat = x_coarse.flatten(2).transpose(1, 2)  # [B, N_coarse, C]
#         x_coarse_flat = self.sr2_norm(x_coarse_flat)
#         # Generate K2, V2
#         k2, v2 = self.kv2(x_coarse_flat).chunk(2, dim=-1)  # Each: [B, N_coarse, C]

#         # 3. Concatenation (多尺度拼接)
#         # K_total: [B, N_fine + N_coarse, C]
#         k_cat = torch.cat([k1, k2], dim=1)
#         v_cat = torch.cat([v1, v2], dim=1)

#         # Reshape for Multi-Head
#         # [B, Heads, (N_fine+N_coarse), d]
#         k = k_cat.reshape(B, -1, self.num_heads, self.d).permute(0, 2, 1, 3)
#         v = v_cat.reshape(B, -1, self.num_heads, self.d).permute(0, 2, 1, 3)

#         # # 4. Attention Calculation
#         # # attn: [B, Heads, N, (N_fine+N_coarse)]
#         # # Q (全图) 去查询 K (多尺度锚点)
#         # attn = (q @ k.transpose(-2, -1)) * self.scale
#         # attn = attn.softmax(dim=-1)

#         # # 保存 Attention Map 用于可视化 (仅保存第一个 Batch 的第一个 Head)
#         # if not self.training:
#         #     self.last_attn_map = attn[0, 0].detach().cpu()

#         # # 5. Weighted Sum
#         # # [B, Heads, N, d]
#         # x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)

#         # 4. Attention Calculation (使用 PyTorch 2.0 官方优化)
#         # 自动实现 FlashAttention 或 Memory-Efficient Attention，不生成庞大的中间矩阵
#         x_out = F.scaled_dot_product_attention(q, k, v)
        
#         # 提取可视化 Attention Map (如果非训练模式)
#         # 由于 SDPA 默认不返回 attention map，可视化时我们需要手动算一下小 batch
#         if not self.training:
#             with torch.no_grad():
#                 # 仅抽取第一张图的第一个头，避免 OOM
#                 q_vis = q[0:1, 0:1, :, :]
#                 k_vis = k[0:1, 0:1, :, :]
#                 attn_map = (q_vis @ k_vis.transpose(-2, -1)) * self.scale
#                 self.last_attn_map = attn_map.softmax(dim=-1).detach().cpu()
        
#         # 5. Reshape back
#         x_out = x_out.transpose(1, 2).reshape(B, N, C)

#         # 6. Final Project
#         x_out = self.proj(x_out)

#         # Reshape back to [B, C, H, W] for CNN
#         return x_out.transpose(1, 2).reshape(B, C, H, W) + x  # Residual Connection
class FG_SRA(nn.Module):        # v2
    """
    Frequency-Guided Spatial Reduction Attention
    【升级 2】频域提取前置：在降采样前注入高频特征
    【升级 3】引入 CPEG：利用 DWConv 找回局部连续性
    """
    def __init__(self, dim, num_heads, sr_ratios=(8, 16)):
        super(FG_SRA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.d = dim // num_heads
        self.scale = self.d ** -0.5

        # 【升级 2】将频域注入器前置，且不再需要传入死板的 spatial_size
        self.freq_inject = FrequencyInjector(dim)

        self.q = nn.Linear(dim, dim, bias=True)

        # --- Scale 1: Fine Grain ---
        self.sr1_ratio = sr_ratios[0]
        self.sr1_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratios[0], stride=sr_ratios[0])
        self.sr1_norm = nn.LayerNorm(dim)
        self.kv1 = nn.Linear(dim, dim * 2, bias=True)

        # --- Scale 2: Coarse Grain ---
        self.sr2_ratio = sr_ratios[1]
        self.sr2_conv = nn.Conv2d(dim, dim, kernel_size=sr_ratios[1], stride=sr_ratios[1])
        self.sr2_norm = nn.LayerNorm(dim)
        self.kv2 = nn.Linear(dim, dim * 2, bias=True)

        # 【升级 3】CPEG 局部连续增强算子 (极其轻量，0.005M 参数)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.last_attn_map = None

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 1. 【高频前置】在全分辨率下进行频域指纹注入！
        x_freq = self.freq_inject(x) # [B, C, H, W]

        # 2. 生成 Query (使用已注入高频指纹的特征)
        x_flat = x_freq.flatten(2).transpose(1, 2)      # [B,C,H,W] -> [B,C,H*W] -> [B,H*W,C] <=> [B,L,C] [B,N,64]
        q = self.q(x_flat).reshape(B, N, self.num_heads, self.d).permute(0, 2, 1, 3) # [B,head,L,d]

        # 3. 生成 K, V
        # --- Branch 1 (Fine) ---
        x_fine = self.sr1_conv(x_freq)
        shape_fine = x_fine.shape # 记录形状用于 CPEG
        x_fine_flat = self.sr1_norm(x_fine.flatten(2).transpose(1, 2))
        k1, v1 = self.kv1(x_fine_flat).chunk(2, dim=-1)

        # --- Branch 2 (Coarse) ---
        x_coarse = self.sr2_conv(x_freq)
        shape_coarse = x_coarse.shape
        x_coarse_flat = self.sr2_norm(x_coarse.flatten(2).transpose(1, 2))
        k2, v2 = self.kv2(x_coarse_flat).chunk(2, dim=-1)

        # 4. 【局部增强】对 Value 进行 3x3 深度可分离卷积，找回切块丢失的边缘连续性
        # v1: [B, N_fine, C] -> [B, C, H/8, W/8]
        v1_spatial = v1.transpose(1, 2).reshape(B, C, shape_fine[2], shape_fine[3]).contiguous()
        v1 = self.v_dwconv(v1_spatial).flatten(2).transpose(1, 2)
        
        # v2: [B, N_coarse, C] -> [B, C, H/16, W/16]
        v2_spatial = v2.transpose(1, 2).reshape(B, C, shape_coarse[2], shape_coarse[3]).contiguous()
        v2 = self.v_dwconv(v2_spatial).flatten(2).transpose(1, 2)

        # 5. 拼接多尺度特征
        k_cat = torch.cat([k1, k2], dim=1)
        v_cat = torch.cat([v1, v2], dim=1)

        k = k_cat.reshape(B, -1, self.num_heads, self.d).permute(0, 2, 1, 3)
        v = v_cat.reshape(B, -1, self.num_heads, self.d).permute(0, 2, 1, 3)

        # 6. SDPA 极速注意力计算 (OOM 克星)
        x_out = F.scaled_dot_product_attention(q, k, v)

        # 可视化缓存
        if not self.training:
            with torch.no_grad():
                q_vis, k_vis = q[0:1, 0:1], k[0:1, 0:1]
                attn_map = (q_vis @ k_vis.transpose(-2, -1)) * self.scale
                self.last_attn_map = attn_map.softmax(dim=-1).detach().cpu()

        # 7. Final Project & Residual
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)

        return x_out.transpose(1, 2).reshape(B, C, H, W) + x


# class EncoderStage(nn.Module):      # v1
#     def __init__(self, in_channels, out_channels, resolution, num_heads=8, sra_ratios=(8, 16)):
#         """
#         in_channels: 上一层的输出通道数
#         out_channels: 本层输出通道数
#         resolution: 本层输入的分辨率 (H, W) 元组
#         num_heads: 注意力头数
#         sra_ratios: FG-SRA 的两个缩减倍率 (Fine, Coarse)
#         """
#         super().__init__()

#         # 1. DWT 下采样 (H, W -> H/2, W/2)
#         self.dwt = DWT_Downsampling()

#         # DWT 后分辨率减半
#         self.curr_res = (resolution[0] // 2, resolution[1] // 2)
#         dwt_channels = in_channels * 4

#         # 2. 1x1 卷积投影 (降维)
#         self.project = nn.Sequential(
#             nn.Conv2d(dwt_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU()
#         )

#         # 3. 特征提取 (ResBlocks)
#         self.res_blocks = nn.Sequential(
#             ResBlock(out_channels),
#             ResBlock(out_channels)
#         )

#         # 4. FG-SRA 注意力 (正式装配!)
#         # 注意：这里的 input_resolution 必须是 DWT 之后的分辨率
#         self.attn = FG_SRA(
#             dim=out_channels,
#             num_heads=num_heads,
#             input_resolution=self.curr_res,
#             sr_ratios=sra_ratios
#         )

#     def forward(self, x):
#         # x: [B, C_in, H, W]

#         x = self.dwt(x)  # [B, C*4, H/2, W/2]
#         x = self.project(x)  # [B, C_out, H/2, W/2]
#         x = self.res_blocks(x)
#         x = self.attn(x)  # FG-SRA 增强

#         return x
class EncoderStage(nn.Module):      # v2
    def __init__(self, in_channels, out_channels, num_heads=8, sra_ratios=(8, 16)):
        """
        in_channels: 上一层的输出通道数
        out_channels: 本层输出通道数
        num_heads: FG-SRA 的注意力头数
        sra_ratios: FG-SRA 的两个缩减倍率 (Fine, Coarse)
        """
        super().__init__()

        # 1. 无损下采样 (H, W -> H/2, W/2)
        self.dwt = DWT_Downsampling()

        # 2. 【升级 1】子带解耦投影 (Subband-Decoupled Projection)
        # 低频子带 (LL) 包含基础语义，使用 1x1 卷积调整通道即可
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 高频子带 (LH, HL, HH) 包含核心微弱篡改痕迹！
        # 必须使用 3x3 卷积给予其充分的局部感受野进行精细特征提取
        self.high_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 解耦提取后，再用一个 3x3 卷积将高低频完美融合
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # 3. 局部特征提取 (保持不变，基础夯实)
        self.res_blocks = nn.Sequential(
            ResBlock(out_channels),
            ResBlock(out_channels)
        )

        # 4. FG-SRA 注意力 (现代化装配)
        # 【升级 2】移除了僵硬的 input_resolution 参数，完全适应动态尺寸
        self.attn = FG_SRA(
            dim=out_channels,
            num_heads=num_heads,
            sr_ratios=sra_ratios
        )

    def forward(self, x):
        # x: [B, C_in, H, W]
        C = x.shape[1]
        dwt_out = self.dwt(x) 
        x_ll = dwt_out[:, 0:C, :, :]         # [B, C, H/2, W/2]
        x_high = dwt_out[:, C:, :, :]        # [B, 3C, H/2, W/2]
        feat_ll = self.low_proj(x_ll)        # [B, C_out, H/2, W/2]
        feat_high = self.high_proj(x_high)   # [B, C_out, H/2, W/2]
        x_fused = torch.cat([feat_ll, feat_high], dim=1) # [B, 2*C_out, H/2, W/2]
        x = self.fuse_proj(x_fused)          # [B, C_out, H/2, W/2]

        # 3. 局部上下文增强
        x = self.res_blocks(x)
        
        # 4. 全局与频域多尺度交互
        x = self.attn(x) 

        return x

class DDG_Bridge(nn.Module):
    """
    DDG-Bridge: Dual-Domain Global Bridge
    Bottleneck 层的核心模块，用于在 24x24 低分辨率下进行全图时空交互。
    """

    def __init__(self, dim, resolution):
        super().__init__()
        self.dim = dim
        self.H, self.W = resolution

        # --- Path A: Global Spatial Attention (Standard MHSA) ---
        # 在 bottleneck 处，24x24=576 个 token，标准 Attention 开销很小且效果最好
        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

        # --- Path B: Global Spectral Filter (FFT) ---
        self.norm_freq = nn.LayerNorm(dim)

        # 1. Channel Mixer: 在进频域前先混合通道，解决"跨通道相关性"问题
        self.freq_channel_mix = nn.Conv2d(dim, dim, 1)

        # 2. Global Spectral Filter (Learnable)
        # FFT 后频域图大小: (H, W/2 + 1)
        # 我们使用复数权重进行滤波: complex_out = complex_in * complex_weight
        self.freq_H = self.H
        self.freq_W = self.W // 2 + 1

        # 定义可学习的复数权重 (使用实部和虚部两个 Parameter)
        # Shape: [dim, H, W/2+1, 2]
        self.complex_weight = nn.Parameter(
            torch.randn(dim, self.freq_H, self.freq_W, 2, dtype=torch.float32) * 0.02
        )

        # --- Gated Fusion ---
        # 门控生成器: 将频域特征映射为 0~1 的权重图
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # 最终融合映射
        self.final_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Input: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert H == self.H and W == self.W, f"Resolution mismatch: expected {self.H}x{self.W}, got {H}x{W}"

        # -------------------------------------------------
        # Path A: Spatial Branch (MHSA)
        # -------------------------------------------------
        # [B, C, H, W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm_spatial(x_flat)

        # Standard Self-Attention
        # attn_out: [B, N, C]
        spatial_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)

        # Reshape back: [B, C, H, W]
        spatial_out = spatial_out.transpose(1, 2).view(B, C, H, W)

        # Residual Connection for Spatial Branch
        spatial_out = x + spatial_out

        # -------------------------------------------------
        # Path B: Frequency Branch (Spectral Filter)
        # -------------------------------------------------
        # 1. Channel Mixing (解决跨通道问题)
        x_mixed = self.freq_channel_mix(x)

        with autocast(enabled=False):
            # 将输入强制转为 fp32
            x_mixed_fp32 = x_mixed.float()
            
            # 2. FFT (Real -> Complex)
            x_fft = torch.fft.rfft2(x_mixed_fp32, norm='ortho')
            
            # 3. Spectral Weighting (确保 weight 也是 fp32)
            weight = torch.view_as_complex(self.complex_weight.float().contiguous())
            x_weighted = x_fft * weight
            
            # 4. IFFT (Complex -> Real)
            freq_out_fp32 = torch.fft.irfft2(x_weighted, s=(H, W), norm='ortho')

        freq_out = freq_out_fp32.to(x.dtype)

        # # 2. FFT (Real -> Complex)
        # # x_fft: [B, C, H, W/2+1] (Complex64)
        # x_fft = torch.fft.rfft2(x_mixed, norm='ortho')

        # # 3. Spectral Weighting
        # # 将权重转为复数 tensor
        # weight = torch.view_as_complex(self.complex_weight.contiguous())  # [C, H, W/2+1]

        # # Element-wise multiplication (Broadcast over Batch)
        # x_weighted = x_fft * weight

        # # 4. IFFT (Complex -> Real)
        # freq_out = torch.fft.irfft2(x_weighted, s=(H, W), norm='ortho')  # [B, C, H, W]

        # Residual Connection for Frequency Branch
        freq_out = x + freq_out

        # -------------------------------------------------
        # Gated Fusion
        # -------------------------------------------------
        # 逻辑: 让频域特征去"门控"空间特征
        # 如果频域发现异常 (Gate -> 1)，则放大对应的空间特征

        gate = self.gate_conv(freq_out)  # [B, C, H, W], values in (0, 1)

        # 融合公式: Spatial * Gate + Frequency
        # 这样既保留了频域的全局滤波效果，又利用频域异常增强了空间语义
        out = spatial_out * gate + freq_out

        # Final Projection
        out = self.final_proj(out)

        return out


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        in_channels: 上一层解码输出的通道数
        skip_channels: Encoder 对应层的通道数
        out_channels: 本层输出通道数
        """
        super().__init__()

        # 1. 子带预测层 (Predictor)
        # 输入 dim -> 输出 dim * 4 (为了生成 LL, LH, HL, HH)
        self.subband_pred = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self.pred_bn = nn.BatchNorm2d(out_channels * 4)

        # 2. IDWT 算子
        self.idwt = IDWT_Upsampling()

        # 3. 掩码引导注意力 (Recursive Guidance)
        # 将上一层的 Mask 映射为空间权重
        self.mask_attn_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可学习的缩放系数

        # 4. 融合层 (Fusion)
        # 接收: IDWT特征 + Skip特征
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # 5. 特征精修 (ResBlock)
        self.res_blocks = nn.Sequential(
            ResBlock(out_channels),
            ResBlock(out_channels)
        )

        # 6. 掩码生成头 (Deep Supervision Head)
        self.mask_head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, skip_feat, prev_mask=None):
        """
        x: 上一层的特征 [B, in_C, H, W]
        skip_feat: Encoder 跳跃连接特征 [B, skip_C, H*2, W*2]
        prev_mask: 上一层预测的 Mask [B, 1, H, W] (可为 None)
        """

        # Step 1: 预测子带
        x_pred = self.pred_bn(self.subband_pred(x)).contiguous()  # [B, out_C*4, H, W]

        # Step 2: IDWT 上采样
        # x_up: [B, out_C, H*2, W*2]
        # subbands: list of [LL, LH, HL, HH] tensors
        x_up, subbands = self.idwt(x_pred)

        # Step 3: 递归掩码引导 (Recursive Mask Guidance)
        if prev_mask is not None:
            # 上采样上一层的 Mask (用插值即可，因为它只是权重)
            # prev_mask: [B, 1, H, W] -> [B, 1, H*2, W*2]
            mask_up = F.interpolate(prev_mask, scale_factor=2, mode='bilinear', align_corners=False)

            # 生成注意力图
            attn_map = self.mask_attn_conv(mask_up)

            # 引导特征: 重点关注 Mask 指示的区域
            x_up = x_up * (1 + self.alpha * attn_map)

        # Step 4: Skip Connection 融合
        # 这里的 skip_feat 包含了 DWT 保留的真实高频细节
        cat_feat = torch.cat([x_up, skip_feat], dim=1)
        x_fused = self.fuse_conv(cat_feat)

        # Step 5: 残差精修
        x_out = self.res_blocks(x_fused)

        # Step 6: 生成当前层的精细 Mask
        curr_mask = self.mask_head(x_out)

        # 返回富结构体 (包含调试信息)
        return {
            'out': x_out,  # 传给下一层 Decoder
            'mask': curr_mask,  # 用于计算 Loss
            'subbands': subbands  # 用于可视化 IDWT 预测结果
        }