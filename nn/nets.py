import torch
from nn.modules import MLP, EncBlock, DecBlock
from functional.loss_functions import bce_recon_loss
import torch.nn.functional as F

def MNIST_MLP_Encoder(out_dim: int, layer_norm: bool=True):
    out_actv_fn = torch.nn.LayerNorm(out_dim) if layer_norm else None
    return MLP(in_dim=784, out_dim=out_dim, num_hidden=3, hidden_dim=512, actv_fn=torch.nn.SiLU(), out_actv_fn=out_actv_fn)

def MNIST_MLP_Decoder(in_dim: int):
    return MLP(in_dim=in_dim, out_dim=784, num_hidden=3, hidden_dim=512, actv_fn=torch.nn.SiLU(), out_actv_fn=torch.nn.Sigmoid())


class MNIST_CNN_Encoder(torch.nn.Module):
    def __init__(self, out_dim: int, mlp: bool=True, layer_norm: bool=True):
        super().__init__()
        modules = [torch.nn.Sequential(
            EncBlock(1, 64, 3, 1, 1, pool=True),
            EncBlock(64, 128, 3, 1, 1, pool=True),
            EncBlock(128, 256, 3, 1, 0),
            EncBlock(256, 512, 3, 1, 0),
            EncBlock(512, out_dim, 3, 1, 0),
            torch.nn.Flatten(),
        )]
        if mlp:
            modules.append(MLP(in_dim=out_dim, out_dim=out_dim, num_hidden=2, hidden_dim=512, actv_fn=torch.nn.SiLU(), out_actv_fn=None))
        if layer_norm:
            modules.append(torch.nn.LayerNorm(out_dim))
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class MNIST_CNN_Decoder(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Unflatten(1, (in_dim, 1, 1)),
            DecBlock(in_dim, 512, 3, 1),
            DecBlock(512, 256, 3, 3),
            DecBlock(256, 128, 3, 3),
            DecBlock(128, 64, 2, 1),
            torch.nn.Conv2d(64, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

class MNIST_AE(torch.nn.Module):
    def __init__(self, out_dim: int, mode: str, layer_norm: bool=True):
        super().__init__()
        if mode == 'mlp':
            self.encoder = MNIST_MLP_Encoder(out_dim, layer_norm)
            self.decoder = MNIST_MLP_Decoder(out_dim)
        elif mode == 'cnn':
            self.encoder = MNIST_CNN_Encoder(out_dim, layer_norm)
            self.decoder = MNIST_CNN_Decoder(out_dim)
        else:
            raise ValueError(f'Invalid mode: {mode}')
    
    def infer(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def loss(self, batch, **kwargs):
        x, _ = batch
        x_hat = self.forward(x)
        return {'loss': bce_recon_loss(x_hat, x), 'mse': F.mse_loss(x_hat, x)}