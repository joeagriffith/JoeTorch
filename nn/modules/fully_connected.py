import torch

class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=1024, num_hidden: int=2, hidden_actv_module: torch.nn.Module=torch.nn.SiLU(), out_actv_module: torch.nn.Module=None):
        super().__init__()

        layers = []
        in_features = in_dim
        
        # Add hidden layers
        for _ in range(num_hidden):
            layers.append(torch.nn.Linear(in_features, hidden_dim))
            if hidden_actv_module is not None:
                layers.append(hidden_actv_module)
            in_features = hidden_dim

        # Add output layer
        layers.append(torch.nn.Linear(in_features, out_dim))
        if out_actv_module is not None:
            layers.append(out_actv_module)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, out_shape: tuple=None):
        out = self.net(x.flatten(start_dim=1))
        if out_shape is not None:
            out = out.view(out.size(0), *out_shape)
        return out
