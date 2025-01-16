import torch
import math

def cosine_schedule(base: float, end: float, T: int, warmup: int = 0, flat_end: int = 0):
    method_T = T - warmup - flat_end
    method_vals = end - (end - base) * ((torch.arange(0, method_T, 1) * math.pi / method_T).cos() + 1) / 2
    return build_schedule(method_vals, T, base, end, warmup, flat_end)

def step_schedule(base: float, T: int, milestones: list[int], gamma: float, warmup: int = 0, flat_end: int = 0):
    method_T = T - warmup - flat_end
    method_vals, val, t = torch.tensor([]), base, 0

    for milestone in milestones:
        method_vals = torch.cat([method_vals, torch.ones(milestone - t) * val])
        t = milestone
        val = val * gamma

    if t < method_T:
        method_vals = torch.cat([method_vals, torch.ones(method_T - t) * val])

    return build_schedule(method_vals, T, base, None, warmup, flat_end, milestones, gamma)

def flat_schedule(base: float, T: int, warmup: int = 0, flat_end: int = 0):
    method_T = T - warmup - flat_end
    method_vals = torch.ones(method_T) * base
    return build_schedule(method_vals, T, base, None, warmup, flat_end)

def build_schedule(method_vals: torch.Tensor, T: int, base: float, end: float = None, warmup: int = 0, flat_end: float = 0, milestones: list[int] = None, gamma: float = None):

    # Linear warmup phase, from 0 to base at start of optimisation
    warmup_vals = torch.linspace(0.0, base, warmup+1)[1:] if warmup > 0 else torch.tensor([])

    # Flat end phase, keep the hyperparameter constant at the end
    flat_end_vals = torch.ones(flat_end) * method_vals[-1] if flat_end > 0 else torch.tensor([])

    return torch.cat([warmup_vals, method_vals, flat_end_vals])