import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DCTLayer(nn.Module):

    def __init__(self, height, width):
        super(DCTLayer, self).__init__()
        self.size = height
        self.register_buffer('dct_matrix', self._get_dct_matrix(height))

    def _get_dct_matrix(self, N):
        dct_m = torch.eye(N)
        for k in range(N):
            for i in range(N):
                w = math.sqrt(2 / N)
                if k == 0:
                    w = math.sqrt(1 / N)
                dct_m[k, i] = w * math.cos(math.pi / N * (i + 0.5) * k)
        return dct_m

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.size and W == self.size, f"Input size ({H}x{W}) mismatch with initialized DCT size ({self.size})"
        x_reshaped = x.reshape(-1, H, W)
        t1 = torch.bmm(self.dct_matrix.unsqueeze(0).expand(x_reshaped.size(0), -1, -1), x_reshaped)
        t2 = torch.bmm(t1, self.dct_matrix.t().unsqueeze(0).expand(x_reshaped.size(0), -1, -1))

        return t2.view(B, C, H, W)