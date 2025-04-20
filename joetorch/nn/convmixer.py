import torch.nn as nn

# Adapted from https://github.com/locuslab/convmixer/blob/main/convmixer.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class convmixer(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_dim, depth, kernel_size=None, patch_size=None):
        super().__init__()

        assert len(in_shape) == 3, "in_shape must be a tuple (C, H, W)"
        if not isinstance(out_shape, int):
            assert (len(out_shape) == 3) or (len(out_shape) == 1), "out_shape must be a tuple (C, H, W) or a single integer"

        if patch_size is None:
            patch_size = max(round(7 * in_shape[-1] / 224), 1)
        if kernel_size is None:
            kernel_size = max(round(7 * (in_shape[-1] / patch_size) / 32), 3)

        self.net = nn.Sequential(
        nn.Conv2d(in_shape[0], hidden_dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(hidden_dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim)
                )),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim)
        ) for i in range(depth)],
        )
        self.out_proj = nn.Conv2d(hidden_dim, out_shape[0], 1) if hidden_dim != out_shape[0] else nn.Identity()
        if len(out_shape) == 1:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
            )
        elif len(out_shape) == 3:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((out_shape[1], out_shape[2])),
            )

    def forward(self, x):
        x = self.net(x)
        x = self.out_proj(x)
        x = self.head(x)

        return x