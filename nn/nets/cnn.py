import torch
import torch.nn as nn

class EncBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, pool=False, bn=False, actv_layer=nn.SiLU(), skip_connection=False):
        super().__init__()
        self.net = [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)]
        if pool:
            self.net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if bn:
            self.net.append(nn.BatchNorm2d(out_dim))
        if actv_layer is not None:
            self.net.append(actv_layer)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        y = self.net(x)
        if self.skip_connection:
            if self.pool:
                skip = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                skip = x
            y += skip
        return y

class DecBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, upsample=False, bn=False, actv_layer=nn.SiLU(), skip_connection=False):
        super().__init__()
        self.net = [nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride)]
        if upsample:
            self.net.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        if bn:
            self.net.append(nn.BatchNorm2d(out_dim))
        if actv_layer is not None:
            self.net.append(actv_layer)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        y = self.net(x)
        if self.skip_connection:
            if self.upsample:
                skip = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            else:
                skip = x
            y += skip
        return y
    

# ============================= Pre-defined Models =============================

class MNIST_CNN_Encoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            EncBlock(1, 64, 3, 1, 1, pool=True),
            EncBlock(64, 128, 3, 1, 1, pool=True),
            EncBlock(128, 256, 3, 1, 0),
            EncBlock(256, 512, 3, 1, 0),
            EncBlock(512, out_dim, 3, 1, 0),
        )
    def forward(self, x):
        return self.net(x).flatten(start_dim=1)


class MNIST_CNN_Decoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            DecBlock(in_dim, 512, 3, 1),
            DecBlock(512, 256, 3, 3),
            DecBlock(256, 128, 3, 3),
            DecBlock(128, 64, 2, 1),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(2).unsqueeze(3))