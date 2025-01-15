import torch
import math

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2

def build_schedule(T, base, end=None, schedule='flat', warmup=0, milestones=None, gamma=None):
    # Warmup phase
    warmup_vals = torch.linspace(0.0, base, warmup+1)[1:] if warmup > 0 else torch.tensor([])

    # Decay phase
    if T < warmup:
        decay_vals = torch.tensor([])
    else:
        if schedule == 'flat':
            decay_vals = torch.ones(T - warmup) * base

        elif schedule == 'cosine':
            assert end is not None, '"end: float" must be provided for cosine schedule'
            decay_vals = cosine_schedule(base, end, T - warmup)

        elif schedule == 'step':
            assert milestones is not None, '"milestones: list[int]" must be provided for step schedule'
            assert gamma is not None, '"gamma: float" must be provided for step schedule'
            decay_vals = torch.tensor([])
            t = warmup
            val = base
            for milestone in milestones:
                decay_vals = torch.cat([decay_vals, torch.ones(milestone - t) * val])
                t = milestone
                val = val * gamma
            if t < T:
                decay_vals = torch.cat([decay_vals, torch.ones(T - t) * val])

        else:
            raise ValueError(f'Unknown learning rate schedule: {schedule}')

    return torch.cat([warmup_vals, decay_vals])[:T]


def get_optimiser(model, optim='AdamW', betas=(0.9, 0.999), momentum=0.9, lr=99999.9, wd=99999.9):

    non_decay_parameters = []
    decay_parameters = []   
    for n, p in model.named_parameters():
        if 'bias' in n or 'bn' in n:
            non_decay_parameters.append(p)
        else:
            decay_parameters.append(p)
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    decay_parameters = [{'params': decay_parameters}]

    if optim == 'AdamW':
        optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd, betas=betas)
    elif optim == 'SGD':
        optimiser = torch.optim.SGD(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd, momentum=momentum)
    
    return optimiser

def apply_weight_decay_with_prior(model, state_dict, weight_decay, fisher=None):
    """
    Apply weight decay to the model parameters towards the given state dict.

    Args:
        model: The model whose parameters will be decayed.
        state_dict: The state dict to be used as the prior.
        weight_decay: The weight decay value to be applied.
    """

    if fisher is None:
        fisher = {name: torch.ones_like(param) for name, param in model.named_parameters()}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.grad += -(state_dict[name].data - param.data) * weight_decay * (fisher[name] / fisher[name].mean())
