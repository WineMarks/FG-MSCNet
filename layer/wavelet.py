import torch
from torch import nn

class DWT_Downsampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_LH = x1 + x2 - x3 - x4
        x_HL = x1 - x2 + x3 - x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)


class IDWT_Upsampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C_4, H, W = x.shape
        C = C_4 // 4
        x_LL = x[:, 0:C, :, :].contiguous()
        x_LH = x[:, C:C * 2, :, :].contiguous()
        x_HL = x[:, C * 2:C * 3, :, :].contiguous()
        x_HH = x[:, C * 3:C * 4, :, :].contiguous()
        subbands = [x_LL, x_LH, x_HL, x_HH]

        x1 = (x_LL + x_LH + x_HL + x_HH) / 2.0
        x2 = (x_LL + x_LH - x_HL - x_HH) / 2.0 
        x3 = (x_LL - x_LH + x_HL - x_HH) / 2.0
        x4 = (x_LL - x_LH - x_HL + x_HH) / 2.0
        
        output = torch.zeros(B, C, H * 2, W * 2, device=x.device, dtype=x.dtype)
        output[:, :, 0::2, 0::2] = x1
        output[:, :, 0::2, 1::2] = x2
        output[:, :, 1::2, 0::2] = x3
        output[:, :, 1::2, 1::2] = x4
        return output, subbands