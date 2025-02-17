import torch
import torch.nn as nn
import torch.nn.functional as F

class EncBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, pool=False, bn=False, dropout=0.0, actv_layer=torch.nn.SiLU(), skip_connection=False):
        super().__init__()
        self.pool = pool
        self.bn = bn
        self.skip_connection = skip_connection

        modules = [torch.nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)]
        if pool:
            modules.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        if bn:
            modules.append(torch.nn.BatchNorm2d(out_dim))
        if dropout > 0.0:
            modules.append(torch.nn.Dropout2d(dropout))
        if actv_layer is not None:
            modules.append(actv_layer)
        self.net = torch.nn.Sequential(*modules)
    
    def forward(self, x):
        y = self.net(x)
        if self.skip_connection:
            if self.pool:
                skip = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                skip = x
            y += skip
        return y


class DecBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, bn=False, actv_layer=torch.nn.SiLU(), dropout=0.0, scale=1.0):
        super().__init__()
        assert scale >= 1.0, "scale must be greater than or equal to 1.0"


        bn = nn.BatchNorm2d(in_dim) if bn else None
        actv_layer = actv_layer if actv_layer is not None else nn.Identity()
        dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None
        convt = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding)
        upsample = nn.Upsample(scale_factor=scale, mode='bilinear') if scale > 1.0 else None

        modules = []
        if bn is not None:
            modules.append(bn)
        if actv_layer is not None:
            modules.append(actv_layer)
        if dropout is not None:
            modules.append(dropout)

        modules.append(convt)

        if upsample is not None:
            modules.append(upsample)

        self.net = torch.nn.Sequential(*modules)
    
    def forward(self, x):
        return self.net(x)

        
class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, gn_groups: int, actv_layer: nn.Module = nn.SiLU(), dropout: float = 0.0, do_residual: bool = True, scale: float = 1.0, num_layers: int = 2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.do_gn = gn_groups > 0
        self.do_actv = actv_layer is not None
        self.do_dropout = dropout > 0.0
        self.do_residual = do_residual
        self.scale = scale
        self.num_layers = num_layers

        assert num_layers > 0, "num_layers must be greater than 0"
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)])
        for _ in range(num_layers-1):
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        if self.do_gn:
            self.groupnorms = nn.ModuleList([nn.GroupNorm(gn_groups, in_channels)])
            for _ in range(num_layers-1):
                self.groupnorms.append(nn.GroupNorm(gn_groups, out_channels))
        
        if self.do_dropout:
            self.dropout = nn.Dropout2d(dropout)

        if self.do_actv:
            self.actv_fn = actv_layer

        if self.do_residual:
            if in_channels == out_channels:
                self.residual_layer = nn.Identity()
            else:
                self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        if scale < 1.0:
            stride = int(1 / scale)
            self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)
        elif scale > 1.0:
            self.up = nn.Upsample(scale_factor=scale)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)

        if self.scale < 1.0:
            # Downsample input
            x = self.down(x)

        # Save input for residual connection
        if self.do_residual:
            residual = x
        
        # Apply layers
        for i in range(self.num_layers):
            # Groupnorm
            if self.do_gn:
                x = self.groupnorms[i](x)
            # Activation
            if self.do_actv:
                x = self.actv_fn(x)
            # Dropout
            if self.do_dropout:
                x = self.dropout(x)
            # Convolution
            x = self.convs[i](x)

        if self.do_residual:
            # Add residual connection
            x = x + self.residual_layer(residual)

        if self.scale > 1.0:
            # Upsample
            x = self.up(x)

        return x

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