import torch
import torch.nn.functional as F

def feature_std(x):
    """
    Compute the standard deviation of the features.
    Args:
        x: torch.Tensor, the input tensor.
    Returns:
        torch.Tensor, the computed standard deviation.
    """
    # x: (N, C)
    return x.std(dim=0).mean()

def feature_std_entropy(x):
    """
    Compute the entropy of the standard deviations of the features.
    Args:
        x: torch.Tensor, the input tensor.
    Returns:
        torch.Tensor, the computed entropy.
    """
    # x: (N, C)
    stds = x.std(dim=0) # stds: (C,)
    normalised = torch.softmax(stds, dim=0) # normalised: (C,)
    entropy = -torch.sum(normalised * torch.log(normalised + 1e-8)) # entropy: ([])
    return entropy

def feature_correlation_matrix(x, standardise:bool=True, unbiased:bool=True):
    """
    Compute the correlation matrix of the features.
    Args:
        x: torch.Tensor, the input tensor.
        standardise: bool, whether to standardise the features.
        unbiased: bool, whether to use the unbiased estimation of the correlation matrix. Use unbiased=True when evaluating encoder or population, and unbiased=False when evaluating the batch.
    Returns:
        torch.Tensor, the computed correlation matrix.
    """
    # x: (N, C)
    if x.size(0) <= 1:
        return torch.tensor(0.0, device=x.device)
    
    # Standardise features, so each feature has mean 0 and std 1
    if standardise:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=True)
        x = (x - mean) / (std + 1e-8)
    
    # Compute normalised auto-correlation
    # Using unbiased estimation (N-1)
    return torch.matmul(x.T, x) / (x.size(0) - 1)

def interfeature_correlation(x, standardise:bool=True, unbiased:bool=True, return_matrix:bool=False):
    """
    Compute the average absolute interfeature correlation of the given input tensor, ignoring self-correlation (where i == j).
    Args:
        x: torch.Tensor, the input tensor.
        standardise: bool, whether to standardise the features.
        unbiased: bool, whether to use the unbiased estimation of the correlation matrix. Use unbiased=True when evaluating encoder or population, and unbiased=False when evaluating the batch.
        return_matrix: bool, whether to return the correlation matrix.
    Returns:
        torch.Tensor, the computed interfeature correlation.
    """

    corr_matrix = feature_correlation_matrix(x, standardise, unbiased)
    
    # Take absolute values since we care about correlation strength
    # regardless of direction
    corrs = corr_matrix.abs()
    
    # select above diagonal to avoid double counting and self-correlation
    corrs = torch.triu(corrs, diagonal=1)
    
    # If there's only one feature, return 0
    if x.size(1) <= 1:
        return torch.tensor(0.0, device=x.device)
    
    # average, accounting for the number of elements above diagonal
    n_elements = (x.size(1) * (x.size(1) - 1)) // 2
    if return_matrix:
        return corrs, corr_matrix
    else:
        return corrs.sum() / n_elements

def get_representations(model, dataset, amp:bool=True):
    device = next(model.parameters()).device
    batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    labels = []

    for x, y in dataloader:
        x = x.to(device)
        with torch.amp.autocast(device.type, enabled=amp, dtype=torch.bfloat16):
            emb = model.embed(x)
        embeddings.append(emb.detach())
        labels.append(y)
    
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    
    return embeddings, labels