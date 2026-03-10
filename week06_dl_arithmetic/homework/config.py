from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 12
    intermediate_dim: int = 4096
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    
    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
