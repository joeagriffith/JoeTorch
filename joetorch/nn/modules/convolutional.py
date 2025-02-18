import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from joetorch.nn.modules.attention import SelfAttention, CrossAttention



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



class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, gn_groups: int=32, actv_fn: nn.Module = nn.SiLU()):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(gn_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(gn_groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.actv_fn = actv_fn

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)

        residual = x

        x = self.groupnorm_1(x)
        x = self.actv_fn(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = self.actv_fn(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)



class ConvSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, gn_groups: int=32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(gn_groups, in_channels)
        self.attention = SelfAttention(1, in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (Batch_Size, Channel, Height, Width)
        residual = x

        x = self.groupnorm(x)
        
        n, c, h, w = x.shape
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height*Width, Channel)
        x = x.view(n, c, h*w).permute(0, 2, 1).contiguous()

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Height*Width, Channel)
        x = self.attention(x)

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Channel, Height, Width)
        x = x.permute(0, 2, 1).contiguous().view(n, c, h, w)

        return x + residual



class ConvCrossAttentionBlock(nn.Module):
    def __init__(self, q_channels: int, kv_channels: int, gn_groups: int=32):
        super().__init__()
        self.q_groupnorm = nn.GroupNorm(gn_groups, q_channels)
        self.kv_groupnorm = nn.GroupNorm(gn_groups, kv_channels)
        self.attention = CrossAttention(1, q_channels, kv_channels)
    
    def forward(self, queries: torch.Tensor, keys_n_values: torch.Tensor) -> torch.Tensor:

        # x: (Batch_Size, Channel, Height, Width)
        residual = queries

        queries = self.q_groupnorm(queries)
        keys_n_values = self.kv_groupnorm(keys_n_values)
        
        n1, c1, h1, w1 = queries.shape
        n2, c2, h2, w2 = keys_n_values.shape
        assert n1 == n2, "Batch size must be the same"

        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height*Width, Channel)
        queries = queries.view(n1, c1, h1*w1).permute(0, 2, 1).contiguous()
        keys_n_values = keys_n_values.view(n2, c2, h2*w2).permute(0, 2, 1).contiguous()

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Height*Width, Channel)
        x = self.attention(queries, keys_n_values)

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Channel, Height, Width)
        x = x.permute(0, 2, 1).contiguous().view(n1, c1, h1, w1)

        return x + residual


