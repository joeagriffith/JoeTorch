import torch
import torch.nn.functional as F

def smooth_l1_loss(input:torch.Tensor, target:torch.Tensor, beta:float=1.0):
    """
    Compute the smooth L1 loss for the given input and target tensors.
    Args:
        input: torch.Tensor, the input tensor.
        target: torch.Tensor, the target tensor.
        beta: float, the beta parameter.
    Returns:
        torch.Tensor, the computed loss.
    """
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()

def negative_cosine_similarity(x1:torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return -torch.matmul(x1, x2.T).sum(dim=-1).mean()

def nll_loss(x:torch.Tensor, target:torch.Tensor):
    return -torch.log_softmax(x, dim=-1).gather(dim=-1, index=target.unsqueeze(1)).squeeze(1).mean()

def smooth_nll_loss(x:torch.Tensor, target:torch.Tensor, beta:float=0.1):
    log_probs = torch.log_softmax(x, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    mean_log_probs = log_probs.mean(dim=-1)
    return nll_loss + beta * mean_log_probs

def mse_recon_loss(x_hat:torch.Tensor, x:torch.Tensor):
    dims = tuple(range(1, len(x.shape)))
    return F.mse_loss(x_hat, x, reduction='none').sum(dims).mean()

def bce_recon_loss(x_hat:torch.Tensor, x:torch.Tensor):
    dims = tuple(range(1, len(x.shape)))
    return F.binary_cross_entropy_with_logits(x_hat, x, reduction='none').sum(dims).mean()

def kl_loss(mu:torch.Tensor, logvar:torch.Tensor):
    dims = tuple(range(1, len(mu.shape)))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dims).mean()

def vae_loss(x_hat:torch.Tensor, x:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor, beta:float=1.0):
    recon_loss = bce_recon_loss(x_hat, x)
    kl_loss = kl_loss(mu, logvar)
    return recon_loss + beta * kl_loss
