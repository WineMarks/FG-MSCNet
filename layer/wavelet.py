import torch
from torch import nn

class DWT_Downsampling(nn.Module):
    def __init__(self):
        super().__init__()
        # DWT本身没有可学习参数，它只是数学变换

    def forward(self, x):
        # x shape: [B, C, H, W]
        # 也就是左上、右上、左下、右下四个像素点
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # Haar Wavelet 公式
        x_LL = x1 + x2 + x3 + x4
        x_LH = x1 + x2 - x3 - x4
        x_HL = x1 - x2 + x3 - x4
        x_HH = x1 - x2 - x3 + x4

        # 拼接在一起: [B, 4*C, H/2, W/2]
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)


class IDWT_Upsampling(nn.Module):
    """
    Inverse Discrete Wavelet Transform (IDWT)
    执行 Haar 小波逆变换，将 4 个子带重组为高分辨率特征图。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Input: [B, C*4, H, W] (假设输入已经包含了LL, LH, HL, HH四个拼接的分量)
        Output: [B, C, H*2, W*2]
        """
        B, C_4, H, W = x.shape
        C = C_4 // 4

        # 1. 切分四个子带
        # Encoder 是 cat([LL, LH, HL, HH])，这里按顺序拆开
        x_LL = x[:, 0:C, :, :].contiguous()
        x_LH = x[:, C:C * 2, :, :].contiguous()
        x_HL = x[:, C * 2:C * 3, :, :].contiguous()
        x_HH = x[:, C * 3:C * 4, :, :].contiguous()

        # 2. 保存子带用于调试可视化 (Optional, return as separate dict if needed)
        subbands = [x_LL, x_LH, x_HL, x_HH]

        # 3. IDWT 逆变换公式 (对应 Encoder 的 Haar 实现)
        # 这里的系数需要和 Encoder 的 DWT 对应。
        # 如果 Encoder 是 sum/diff，Decoder 需要除以 2 或 1 来还原幅度。
        # 假设 Encoder 没做归一化 (直接相加减)，这里我们除以 2 来恢复原始像素幅度。
        # x1 = (LL + LH + HL + HH) / 2 ...

        x1 = (x_LL + x_LH + x_HL + x_HH) / 2.0  # Top-Left
        x2 = (x_LL + x_LH - x_HL - x_HH) / 2.0  # Top-Right
        x3 = (x_LL - x_LH + x_HL - x_HH) / 2.0  # Bottom-Left
        x4 = (x_LL - x_LH - x_HL + x_HH) / 2.0  # Bottom-Right

        # 4. 像素重排 (Interleave / Pixel Shuffle)
        # 我们需要把 x1, x2, x3, x4 按 2x2 格子拼回去

        output = torch.zeros(B, C, H * 2, W * 2, device=x.device, dtype=x.dtype)

        output[:, :, 0::2, 0::2] = x1
        output[:, :, 0::2, 1::2] = x2
        output[:, :, 1::2, 0::2] = x3
        output[:, :, 1::2, 1::2] = x4

        return output, subbands