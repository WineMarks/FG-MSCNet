import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .modules import InputStem, EncoderStage, DecoderStage, DDG_Bridge


class FG_MSCNet(nn.Module):
    def __init__(self, num_classes=1, img_size=384):
        super(FG_MSCNet, self).__init__()

        self.img_size = img_size

        # --- 1. Input Stem (RGB + SRM Fusion) ---
        # Output: [B, 64, H, W]
        self.stem = InputStem(in_channels=3, srm_channels=30, base_channels=64)

        # --- 2. Encoder (DWT + FG-SRA) ---
        # Stage 1: In(64) -> DWT -> Proj(64) -> Out(64, H/2, W/2)
        self.enc1 = EncoderStage(in_channels=64, out_channels=64,
                                 sra_ratios=(8, 16))

        # Stage 2: In(64) -> DWT -> Proj(128) -> Out(128, H/4, W/4)
        self.enc2 = EncoderStage(in_channels=64, out_channels=128,
                                 sra_ratios=(4, 8))

        # Stage 3: In(128) -> DWT -> Proj(256) -> Out(256, H/8, W/8)
        self.enc3 = EncoderStage(in_channels=128, out_channels=256,
                                 sra_ratios=(2, 4))

        # Stage 4: In(256) -> DWT -> Proj(512) -> Out(512, H/16, W/16)
        self.enc4 = EncoderStage(in_channels=256, out_channels=512,
                                 sra_ratios=(1, 2))

        # --- 3. Bottleneck (DDG-Bridge) ---
        # Global Frequency-Spatial Interaction
        self.bridge = DDG_Bridge(dim=512, resolution=(img_size // 16, img_size // 16))

        # --- 4. Decoder (IDWT + Recursive Refinement) ---
        # Dec 1: Upsample to H/8. Skip from Enc3 (256).
        # In(512) -> IDWT(256) + Skip(256) -> Out(256)
        self.dec1 = DecoderStage(in_channels=512, skip_channels=256, out_channels=256)

        # Dec 2: Upsample to H/4. Skip from Enc2 (128).
        # In(256) -> IDWT(128) + Skip(128) -> Out(128)
        self.dec2 = DecoderStage(in_channels=256, skip_channels=128, out_channels=128)

        # Dec 3: Upsample to H/2. Skip from Enc1 (64).
        # In(128) -> IDWT(64) + Skip(64) -> Out(64)
        self.dec3 = DecoderStage(in_channels=128, skip_channels=64, out_channels=64)

        # Dec 4: Upsample to H. Skip from Stem (64).
        # In(64) -> IDWT(32) + Skip(64) -> Out(32)
        self.dec4 = DecoderStage(in_channels=64, skip_channels=64, out_channels=32)

        # --- 【新增】全局自适应校准头 (Adaptive Calibration Head) ---
        # 接收 Bridge 层 (512通道) 的最深层全局特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_bias_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),  # <--- 新增内生约束
            nn.GELU(),
            nn.Linear(128, 1) # 输出一个不受限的标量 Bias
        )

        nn.init.constant_(self.adaptive_bias_head[-1].weight, 0)
        nn.init.constant_(self.adaptive_bias_head[-1].bias, 0)

        # --- 5. Final Output Head ---
        self.final_head = nn.Conv2d(32, num_classes, kernel_size=1)

        # Weight Initialization
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
        """
        处理任意分辨率输入：确保输入是 32 的倍数 (16 * 2, DWT 需要偶数切分)
        """
        H, W = x.shape[2], x.shape[3]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, H, W

    def forward(self, x, return_features=False):
        # 1. Dynamic Padding
        x_padded, original_H, original_W = self.check_input_size(x)

        # --- Encoder Path ---
        # Stem
        # stem_feat = self.stem(x_padded)  # [B, 64, H, W]
        stem_feat = checkpoint(self.stem, x_padded, use_reentrant=False)

        # # Stage 1
        # enc1_feat = self.enc1(stem_feat)  # [B, 64, H/2, W/2]

        # # Stage 2
        # enc2_feat = self.enc2(enc1_feat)  # [B, 128, H/4, W/4]

        # # Stage 3
        # enc3_feat = self.enc3(enc2_feat)  # [B, 256, H/8, W/8]

        # # Stage 4
        # enc4_feat = self.enc4(enc3_feat)  # [B, 512, H/16, W/16]
        enc1_feat = checkpoint(self.enc1, stem_feat, use_reentrant=False)
        enc2_feat = checkpoint(self.enc2, enc1_feat, use_reentrant=False)
        enc3_feat = checkpoint(self.enc3, enc2_feat, use_reentrant=False)
        enc4_feat = checkpoint(self.enc4, enc3_feat, use_reentrant=False)

        # --- Bottleneck ---
        # bridge_feat = self.bridge(enc4_feat)  # [B, 512, H/16, W/16]
        bridge_feat = checkpoint(self.bridge, enc4_feat, use_reentrant=False)

        # --- 【新增】提取全局特征，预测当前图像的动态偏移量 ---
        global_feat = self.global_pool(bridge_feat).flatten(1) # [B, 512]
        # bias: [B, 1, 1, 1] 方便与 [B, 1, H, W] 的空间掩码直接相加
        adaptive_bias = self.adaptive_bias_head(global_feat).view(-1, 1, 1, 1)

        # --- Decoder Path (Deep Supervision) ---
        # Mask 容器，用于存放多尺度预测结果
        masks = []

        # Dec 1 (H/16 -> H/8)
        # 输入: Bridge特征, Enc3跳跃连接, 上一层Mask(None)
        # dec1_out = self.dec1(bridge_feat, enc3_feat, prev_mask=None)
        # masks.append(dec1_out['mask'])  # Mask 1 (Coarse)

        # # Dec 2 (H/8 -> H/4)
        # dec2_out = self.dec2(dec1_out['out'], enc2_feat, prev_mask=dec1_out['mask'])
        # masks.append(dec2_out['mask'])  # Mask 2

        # # Dec 3 (H/4 -> H/2)
        # dec3_out = self.dec3(dec2_out['out'], enc1_feat, prev_mask=dec2_out['mask'])
        # masks.append(dec3_out['mask'])  # Mask 3

        # # Dec 4 (H/2 -> H)
        # # 注意：这里 Skip Connection 用的是 Stem 特征 (Full Resolution)
        # dec4_out = self.dec4(dec3_out['out'], stem_feat, prev_mask=dec3_out['mask'])
        dec1_out = checkpoint(self.dec1, bridge_feat, enc3_feat, None, use_reentrant=False)
        masks.append(dec1_out['mask'])
        
        dec2_out = checkpoint(self.dec2, dec1_out['out'], enc2_feat, dec1_out['mask'], use_reentrant=False)
        masks.append(dec2_out['mask'])
        
        dec3_out = checkpoint(self.dec3, dec2_out['out'], enc1_feat, dec2_out['mask'], use_reentrant=False)
        masks.append(dec3_out['mask'])
        
        dec4_out = checkpoint(self.dec4, dec3_out['out'], stem_feat, dec3_out['mask'], use_reentrant=False)
        # Mask 4 由 Final Head 生成，这里先不加进列表，或者作为最精细的Mask

        # --- Final Head ---
        final_mask = self.final_head(dec4_out['out'])  # [B, 1, H, W]

        final_mask = final_mask + adaptive_bias

        masks = [m + adaptive_bias for m in masks]

        # 裁剪回原始尺寸 (如果是 Padding 过的)
        if final_mask.shape[2] != original_H or final_mask.shape[3] != original_W:
            final_mask = final_mask[:, :, :original_H, :original_W]
            # 同时也裁剪 Deep Supervision 的 masks
            for i in range(len(masks)):
                scale = 2 ** (3 - i)  # 计算对应的下采样倍率
                masks[i] = masks[i][:, :, :original_H // scale, :original_W // scale]

        # 组织输出
        if return_features:
            return {
                'mask_final': final_mask,
                'masks_aux': masks,  # [Mask_H/8, Mask_H/4, Mask_H/2]
                'stem_noise': stem_feat,  # 用于可视化 SRM 提取效果
                'freq_gate': self.bridge.gate_conv[1].output if hasattr(self.bridge, 'gate_conv') else None,
                # 这一步需要hook或者在bridge里保存
                'dec_subbands': dec4_out['subbands'],  # 查看最后一层 IDWT 重建的高频分量
                'adaptive_bias': adaptive_bias # 加入字典方便后续在 TensorBoard 里监控
            }
        else:
            # 训练时通常只需要 Mask 列表
            return [final_mask, *reversed(masks)]
            # 返回顺序: [Final(H), Mask3(H/2), Mask2(H/4), Mask1(H/8)]


# Quick Sanity Check
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