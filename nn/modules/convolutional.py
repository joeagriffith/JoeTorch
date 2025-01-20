import torch

class EncBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, pool=False, bn=False, actv_layer=torch.nn.SiLU(), skip_connection=False):
        super().__init__()
        self.pool = pool
        self.bn = bn
        self.skip_connection = skip_connection

        modules = [torch.nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)]
        if pool:
            modules.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        if bn:
            modules.append(torch.nn.BatchNorm2d(out_dim))
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
    def __init__(self, in_dim, out_dim, kernel_size, stride, upsample=False, bn=False, actv_layer=torch.nn.SiLU(), skip_connection=False):
        super().__init__()
        self.upsample = upsample
        self.bn = bn
        self.skip_connection = skip_connection

        modules = [torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride)]
        if upsample:
            modules.append(torch.nn.Upsample(scale_factor=2, mode='bilinear'))
        if bn:
            modules.append(torch.nn.BatchNorm2d(out_dim))
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
    
