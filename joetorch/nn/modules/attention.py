import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import scaled_dot_product_attention

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        assert d_embed % n_heads == 0, f"d_embed {d_embed} must be divisible by n_heads {n_heads}"

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: (Batch_size, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape        
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, 3*Dim) -> 3 x (Batch_size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) 

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, n_heads, d_head) -> (Batch_size, n_heads, Seq_Len, d_head)
        q = q.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        k = k.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        v = v.view(*interim_shape).permute(0, 2, 1, 3).contiguous()

        # Use flash attention via scaled_dot_product_attention
        output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal_mask
        )

        # (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, Seq_Len, n_heads, d_head) -> (Batch_size, Seq_Len, Dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(*input_shape)

        output = self.out_proj(output)

        # (Batch_size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        assert d_embed % n_heads == 0, f"d_embed {d_embed} must be divisible by n_heads {n_heads}"

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.kv_proj = nn.Linear(d_cross, d_embed*2, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    
    def forward(self, query, keys_n_values):
        # query: (Batch_size, Seq_Len_Q, Dim_Q)
        # keys_n_values: (Batch_size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = query.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project inputs
        q = self.q_proj(query)
        kv = self.kv_proj(keys_n_values)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        k = k.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        v = v.view(*interim_shape).permute(0, 2, 1, 3).contiguous()

        # Use flash attention via scaled_dot_product_attention
        output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        output = output.permute(0, 2, 1, 3).contiguous().view(*input_shape)
        output = self.out_proj(output)
        return output
