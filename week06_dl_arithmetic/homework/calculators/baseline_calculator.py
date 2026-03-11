"""
Baseline Calculator for DDP implementation with baseline model.
"""

from calculators.base import BaseCalculator


class BaselineCalculator(BaseCalculator):
    """
    Calculator for baseline implementation with DDP.
    """
    
    def calculate_total_params(self) -> int:
        """
        Calculate total model parameters.
        """
        v = self.model.vocab_size
        h = self.model.hidden_dim
        l = self.model.num_layers
        i = self.model.intermediate_dim

        embedding = v * h
        lm_head = h * v
        per_layer = 4 * h * h + 3 * h * i + 2 * h
        final_norm = h

        return embedding + l * per_layer + final_norm + lm_head
    
    def calculate_param_memory(self) -> int:
        """
        DDP: full params on each GPU.

        With AMP: params in bf16 + master params in fp32
        """
        total_params = self.calculate_total_params()
        bf16_params = total_params * self.training.dtype_bytes
        fp32_master_params = total_params * 4
        return bf16_params + fp32_master_params

    def calculate_gradient_memory(self) -> int:
        """
        DDP: full gradients on each GPU (fp32).
        """
        total_params = self.calculate_total_params()
        return total_params * 4

    def calculate_optimizer_memory(self) -> int:
        """
        DDP: full optimizer states on each GPU (fp32).

        AdEMAMix has 3 states: m (momentum), v (variance), nu (third moment)
        """
        total_params = self.calculate_total_params()
        return total_params * 3 * 4

    def calculate_activation_memory(self) -> int:
        """
        Baseline activation memory (all intermediates saved).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        nh = self.model.num_heads
        l = self.model.num_layers
        i = self.model.intermediate_dim
        v = self.model.vocab_size
        db = self.training.dtype_bytes

        tokens_h = b * s * h
        attn_scores = b * nh * s * s

        per_layer_saved = 6 * tokens_h + 3 * b * s * i + 2 * attn_scores
        embeddings_saved = tokens_h
        logits_saved = b * s * v

        return (embeddings_saved + l * per_layer_saved + logits_saved) * db
    
    def calculate_peak_memory(self) -> int:
        """Total peak memory = params + grads + optimizer + activations."""
        return (
            self.calculate_param_memory()
            + self.calculate_gradient_memory()
            + self.calculate_optimizer_memory()
            + self.calculate_activation_memory()
        )

    def time_embedding_ms(self) -> float:
        """
        Embedding lookup time (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        db = self.training.dtype_bytes

        flops = 0
        memory_bytes = 2 * b * s * h * db
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_rms_norm_ms(self) -> float:
        """
        RMSNorm time - baseline, not fused (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        db = self.training.dtype_bytes

        elements = b * s * h
        flops = 8 * elements
        memory_bytes = (2 * elements + h) * db
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_attention_ms(self) -> float:
        """
        Standard attention time (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        nh = self.model.num_heads
        db = self.training.dtype_bytes

        proj_flops = 8 * b * s * h * h
        attn_flops = 4 * b * s * s * h
        softmax_flops = 5 * b * nh * s * s
        flops = proj_flops + attn_flops + softmax_flops

        attn_scores = b * nh * s * s
        memory_bytes = (6 * b * s * h + 2 * attn_scores) * db
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_mlp_ms(self) -> float:
        """
        MLP time - baseline (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        i = self.model.intermediate_dim
        db = self.training.dtype_bytes

        matmul_flops = 6 * b * s * h * i
        activation_flops = 8 * b * s * i
        flops = matmul_flops + activation_flops

        memory_bytes = (2 * b * s * h + 3 * b * s * i) * db
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_lm_head_ms(self) -> float:
        """
        LM head projection time (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        v = self.model.vocab_size
        db = self.training.dtype_bytes

        flops = 2 * b * s * h * v
        memory_bytes = (b * s * h + h * v + b * s * v) * db
        return self.roofline_time_ms(flops, memory_bytes)
    
    def time_loss_ms(self) -> float:
        """
        Cross-entropy loss time - baseline (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        v = self.model.vocab_size
        db = self.training.dtype_bytes

        flops = 5 * b * s * v
        memory_bytes = 2 * b * s * v * db
        return self.roofline_time_ms(flops, memory_bytes)

    def calculate_communication_volume(self) -> int:
        """
        DDP all-reduce volume (bytes).
        
        all-reduce: 2 * (n-1)/n * gradient_size
        ≈ 2 * gradient_size for large n
        """
        n = self.training.num_gpus
        if n <= 1:
            return 0

        grad_bytes = self.calculate_gradient_memory()
        return int((2 * (n - 1) / n) * grad_bytes)
    
    def time_communication_ms(self) -> float:
        """
        DDP communication time (ms).
        """
        volume = self.calculate_communication_volume()
        bw = self.gpu.interconnect_bandwidth_gbps * 1e9
        return (volume / bw) * 1000
    
    def overlap_efficiency(self) -> float:
        """
        DDP overlap efficiency (0.0 to 1.0).
        
        DDP overlaps gradient all-reduce with backward computation.
        Estimate based on your analysis.
        """
        return 0.7
    
    def time_total_step_ms(self) -> float:
        """
        Total step time accounting for compute/comm overlap (ms).
        
        Consider how to combine compute time and communication time
        with overlap efficiency.
        """
        compute_ms = self.time_forward_backward_ms()
        comm_ms = self.time_communication_ms()
        overlap = self.overlap_efficiency()
        return compute_ms + (1.0 - overlap) * comm_ms
