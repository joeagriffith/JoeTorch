import torch.nn as nn

# Adapted from https://github.com/locuslab/convmixer/blob/main/convmixer.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# defaults for 128x128 images
class ConvMixer(nn.Module):
    def __init__(self, in_dim, out_dim, depth, kernel_size=5, patch_size=4):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(out_dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size, groups=out_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(out_dim)
                )),
                nn.Conv2d(out_dim, out_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(out_dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        )

    def forward(self, x, stop_at=None):
        if stop_at == 0:
            return x
        else: 
            return self.net(x)
        