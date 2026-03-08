"""
Attention with RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn import flash_attn_func

from config import TransformerConfig
# torch.use_deterministic_algorithms(True)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """
    # TODO: Use fused RoPE from flash_attn library instead

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache up to seq_len."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)

        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to q and k.
        
        Args:
            q: (B, num_heads, S, head_dim)
            k: (B, num_heads, S, head_dim)
            seq_len: sequence length (must be <= max_seq_len)
            
        Returns:
            q_rotated, k_rotated with same shapes
        """
        assert seq_len <= self.max_seq_len, \
            f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to tensor x."""
        return apply_rotary_emb(x, cos, sin, max_seqlen=self.max_seq_len)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with vanilla implementation and RoPE.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        # TODO: Replace with fused QKV projection
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)


        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, H = x.shape

        qkv: torch.Tensor = self.qkv_proj(x)

        q = qkv[:, :, 0:self.hidden_dim].view(B, S, self.num_heads, self.head_dim).contiguous()
        k = qkv[:, :, self.hidden_dim:2*self.hidden_dim].view(B, S, self.num_heads, self.head_dim).contiguous()
        v = qkv[:, :, 2*self.hidden_dim:3*self.hidden_dim].view(B, S, self.num_heads, self.head_dim).contiguous()

        q, k = self.rope.forward(q, k, S)


        # TODO: Replace vanilla attention with Flash Attention
        out: torch.Tensor = flash_attn_func(q, k, v, causal=True, deterministic=True, dropout_p=self.config.dropout if self.training else 0.0,)

        out = out.view(B, S, H)
        out = self.out_proj(out)

        return out
