import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=1024, num_hidden: int=2, hidden_actv_module: nn.Module=nn.SiLU(), out_actv_module: nn.Module=None):
        super().__init__()

        layers = []
        in_features = in_dim
        
        # Add hidden layers
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            if hidden_actv_module is not None:
                layers.append(hidden_actv_module)
            in_features = hidden_dim

        # Add output layer
        layers.append(nn.Linear(in_features, out_dim))
        if out_actv_module is not None:
            layers.append(out_actv_module)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, out_shape: tuple=None):
        out = self.net(x.flatten(start_dim=1))
        if out_shape is not None:
            out = out.view(out.size(0), *out_shape)
        return out

# ============================= Pre-defined Models =============================

def MNIST_MLP_Encoder(out_dim: int):
    return MLP(in_dim=784, out_dim=out_dim, num_hidden=3, hidden_dim=512, actv_fn=nn.SiLU(), out_actv_fn=nn.SiLU())

def MNIST_MLP_Decoder(in_dim: int):
    return MLP(in_dim=in_dim, out_dim=784, num_hidden=3, hidden_dim=512, actv_fn=nn.SiLU(), out_actv_fn=nn.Sigmoid())

def MLP_Bottleneck(in_dim: int, out_dim: int=None, layer_norm: bool=True):
    if out_dim is None:
        out_dim = in_dim
    out_actv_fn = nn.LayerNorm(out_dim) if layer_norm else None
    return MLP(in_dim=in_dim, out_dim=out_dim, num_hidden=2, hidden_dim=512, actv_fn=nn.SiLU(), out_actv_fn=out_actv_fn)