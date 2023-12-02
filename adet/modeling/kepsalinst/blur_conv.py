import torch
from torch.nn import functional as F
from torch import nn


class GaussianBlurConv(nn.Module):
    def __init__(self, channels, length, sigma):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = self.generate_kernel(int(length), sigma)
        # kernel = [[0.11, 0.11, 0.11],
        #           [0.11, 0.12, 0.11],
        #           [0.11, 0.11, 0.11]]
        kernel = torch.tensor(kernel)

        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, length, length))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        self.pad = int((length - 1) / 2)

    def forward(self, x):
        x = F.pad(x, pad=(self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = F.conv2d(x, self.weight.to(x.device), padding=0, groups=self.channels)
        return x

    def generate_kernel(self, l, sigma=1):
        assert l % 2 == 1

        c = torch.tensor([(l - 1) / 2, (l - 1) / 2], dtype=torch.int64)

        x_locs = torch.arange(0, l)
        y_locs = torch.arange(0, l)

        mesh_x, mesh_y = torch.meshgrid(x_locs, y_locs)
        mesh = torch.cat([mesh_y.unsqueeze(2), mesh_x.unsqueeze(2)], 2)

        # kernel = torch.exp(-(torch.sum((mesh - c) ** 2, -1)) / (2 * sigma**2)) / (2.507 * sigma)
        kernel = torch.exp(-(torch.sum((mesh - c) ** 2, -1)) / (2 * sigma ** 2))
        return kernel

