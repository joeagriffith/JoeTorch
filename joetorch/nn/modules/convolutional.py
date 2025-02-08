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
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, upsample=False, bn=False, dropout=0.0, actv_layer=torch.nn.SiLU(), skip_connection=False):
        super().__init__()
        self.upsample = upsample
        self.bn = bn
        self.skip_connection = skip_connection

        modules = [torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding)]
        if upsample:
            modules.append(torch.nn.Upsample(scale_factor=2, mode='bilinear'))
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
            if self.upsample:
                skip = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            else:
                skip = x
            y += skip
        return y

        
class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: float = 1.0, dropout: float = 0.0):
        super().__init__()

        self.scale = scale
        if scale < 1.0:
            stride = int(1 / scale)
            self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)
        elif scale > 1.0:
            self.up = nn.Upsample(scale_factor=scale)

        if dropout > 0.0:
            self.do_dropout = True
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.do_dropout = False

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)

        if self.scale < 1.0:
            x = self.down(x)

        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        if self.do_dropout:
            x = self.dropout(x)
        x = self.conv_2(x)

        x = x + self.residual_layer(residual)

        if self.scale > 1.0:
            x = self.up(x)

        return x