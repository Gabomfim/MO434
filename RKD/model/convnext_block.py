import torch
import torch.nn as nn

__all__ = ['ConvNextBlock', 'Downsample']


class DropPath(nn.Module):
    """Stochastic Depth: randomly drops the residual branch per sample."""

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # one mask value per sample (broadcast over C, H, W)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class ConvNextBlock(nn.Module):
    """ConvNeXt block.

    Flow (tensors are NCHW; LayerNorm and the 1x1 convs operate channel-last):

        x ----------------------------------------------------+ (skip)
        |                                                     |
        Depthwise Conv2d 7x7, stride 1, pad 3   (C -> C)      |
        LayerNorm                                             |
        Conv2d 1x1   (C  -> 4C)   [expansion]                 |
        GELU                                                  |
        Conv2d 1x1   (4C -> C)    [projection]                |
        Layer Scale (gamma)                                   |
        Drop Path                                             |
        +-----------------------------------------------------+
        |
        out (H, W, C)
    """

    def __init__(self, channels, drop_path=0.0, layer_scale_init=1e-6):
        super(ConvNextBlock, self).__init__()

        # Depthwise 7x7 conv keeps H, W (stride 1, pad 3) and channels (groups=C).
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, stride=1,
                                padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)

        # 1x1 convs == pointwise / per-pixel linear layers.
        self.pwconv1 = nn.Linear(channels, 4 * channels)   # expansion C -> 4C
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)   # projection 4C -> C

        # Layer Scale: learnable per-channel scaling of the residual branch.
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        ) if layer_scale_init > 0 else None

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        skip = x

        x = self.dwconv(x)
        # NCHW -> NHWC so LayerNorm / Linear act over the channel dimension.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)

        return skip + self.drop_path(x)


class Downsample(nn.Module):
    """ConvNeXt downsampling layer.

    Flow (tensors are NCHW; LayerNorm operates channel-last):

        x  (H,   W,   C_in)
        LayerNorm
        Conv2d 2x2, stride 2   (C_in -> C_out)
        out  (H/2, W/2, C_out)
    """

    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # NCHW -> NHWC so LayerNorm acts over the channel dimension.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # NHWC -> NCHW for the conv.
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x
