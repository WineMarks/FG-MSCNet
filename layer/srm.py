import torch
import torch.nn as nn
import numpy as np


class SRMConv2d_30(nn.Module):
    def __init__(self, in_channels=3):
        super(SRMConv2d_30, self).__init__()
        self.in_channels = in_channels
        srm_kernel = self._get_srm_kernels()  # shape: (30, 1, 5, 5)

        self.srm_conv = nn.Conv2d(in_channels, 30, kernel_size=5, stride=1, padding=2, bias=False)
        srm_kernel = torch.from_numpy(srm_kernel).float()
        self.srm_conv.weight.data = srm_kernel.repeat(1, in_channels, 1, 1) / in_channels
        for param in self.srm_conv.parameters():
            param.requires_grad = False

        self.bn = nn.BatchNorm2d(30)
        self.act = nn.Tanh()

    def forward(self, x):
        x_scaled = x * 255.0
        noise = self.srm_conv(x_scaled)

        return noise

    def _get_srm_kernels(self):
        f1 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, -1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]).astype(np.float32)
        f2 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]).astype(np.float32)
        f3 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [-1, 3, -3, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]).astype(np.float32)
        def _get_rotations(kernel):
            kernels = []
            kernels.append(kernel)
            kernels.append(np.rot90(kernel, 1))
            kernels.append(np.rot90(kernel, 2))
            kernels.append(np.rot90(kernel, 3))
            return kernels
        kernels = []
        k_list = [f3]
        for k in k_list:
            kernels.extend(_get_rotations(k))

        k_list = [f1, f2]
        for k in k_list:
            kernels.extend(_get_rotations(k))
        minmax = np.array([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]]).astype(np.float32) / 12.0
        kernels.append(minmax)
        lap3 = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, -4, 1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0]]).astype(np.float32)
        kernels.append(lap3)
        current_count = len(kernels)
        for i in range(30 - current_count):
            k = np.random.randn(5, 5)
            k[2, 2] = -np.sum(k) + k[2, 2]
            k = k / np.sum(np.abs(k))
            kernels.append(k.astype(np.float32))
        srm_kernels = np.stack(kernels, axis=0)[:, np.newaxis, :, :]
        return srm_kernels