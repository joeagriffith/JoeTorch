import torch.nn as nn
import numpy as np
import math



# code sourced from https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py

def interpolate_pos_embedding(x, pos_embed):
    B, npatch, D = x.shape
    if len(pos_embed.shape) == 2:
        N, _ = pos_embed.shape
    elif len(pos_embed.shape) == 3:
        _, N, _ = pos_embed.shape

    if npatch == N:
        return pos_embed
    dim = x.shape[-1]
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=math.sqrt(npatch / N),
        mode='bicubic',
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, height, width, cls_token=False, normalize=True):
    """
    height: int of the grid height
    width: int of the grid width
    return:
    pos_embed: [height*width, embed_dim] or [1+height*width, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(height, dtype=float)
    grid_w = np.arange(width, dtype=float)
    if normalize:
        grid_h = grid_h / (height - 1)
        grid_w = grid_w / (width - 1)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, height, width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False, normalize=True):
    """
    length: int of the grid length
    return:
    pos_embed: [length, embed_dim] or [1+length, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(length, dtype=float)
    if normalize:
        grid = grid / (length - 1)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
