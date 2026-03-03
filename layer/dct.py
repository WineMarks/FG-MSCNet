import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DCTLayer(nn.Module):
    """
    独立 DCT 模块：执行二维离散余弦变换 (2D-DCT)。
    采用正交归一化的 DCT-II 标准。
    """

    def __init__(self, height, width):
        super(DCTLayer, self).__init__()
        assert height == width, "DCTLayer 目前仅支持方形特征图以保证矩阵计算效率"
        self.size = height

        # 1. 生成 DCT 变换矩阵 (Orthogonal DCT-II)
        # 形状: [size, size]
        self.register_buffer('dct_matrix', self._get_dct_matrix(height))

    def _get_dct_matrix(self, N):
        """生成 N x N 的 DCT 变换矩阵"""
        dct_m = torch.eye(N)
        for k in range(N):
            for i in range(N):
                w = math.sqrt(2 / N)
                if k == 0:
                    w = math.sqrt(1 / N)
                dct_m[k, i] = w * math.cos(math.pi / N * (i + 0.5) * k)
        return dct_m

    def forward(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, C, H, W] (频域谱)
        原理: F = A * X * A^T
        """
        B, C, H, W = x.shape
        assert H == self.size and W == self.size, f"Input size ({H}x{W}) mismatch with initialized DCT size ({self.size})"

        # 矩阵乘法实现 2D DCT: F = M @ X @ M.t()
        # M: [H, H], x: [B*C, H, W]
        x_reshaped = x.reshape(-1, H, W)

        # 1. 对列做 1D-DCT
        t1 = torch.bmm(self.dct_matrix.unsqueeze(0).expand(x_reshaped.size(0), -1, -1), x_reshaped)
        # 2. 对行做 1D-DCT (即乘转置矩阵)
        t2 = torch.bmm(t1, self.dct_matrix.t().unsqueeze(0).expand(x_reshaped.size(0), -1, -1))

        return t2.view(B, C, H, W)