"""
Efficient Calculator for FSDP implementation with efficient_model.
"""

from calculators.base import BaseCalculator


class EfficientCalculator(BaseCalculator):
    """
    Calculator for efficient implementation with FSDP.
    """

    def calculate_total_params(self) -> int:
        """Same as baseline - model architecture unchanged."""
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
        FSDP: sharded params (fp32).
        """
        shard_factor = max(self.training.num_gpus, 1)
        return int(self.calculate_total_params() * 4 / shard_factor)

    def calculate_gradient_memory(self) -> int:
        """
        FSDP: sharded gradients after reduce-scatter (fp32).
        """
        shard_factor = max(self.training.num_gpus, 1)
        return int(self.calculate_total_params() * 4 / shard_factor)

    def calculate_optimizer_memory(self) -> int:
        """
        FSDP: sharded optimizer states (fp32).
        
        AdEMAMix has 3 states: m (momentum), v (variance), nu (third moment)
        """
        shard_factor = max(self.training.num_gpus, 1)
        return int(self.calculate_total_params() * 3 * 4 / shard_factor)

    def calculate_fsdp_buffer_memory(self) -> int:
        """
        FSDP communication buffers (bf16).
        
        - 2 All-gather buffers: unsharded params for current unit
        - 2 Reduce-scatter buffers: gradients before sharding
        """
        h = self.model.hidden_dim
        i = self.model.intermediate_dim
        db = self.training.dtype_bytes

        per_layer_params = 4 * h * h + 3 * h * i + 2 * h
        return 4 * per_layer_params * db

    def calculate_activation_memory(self) -> int:
        """
        Efficient activation memory.
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        i = self.model.intermediate_dim
        l = self.model.num_layers
        db = self.training.dtype_bytes

        tokens_h = b * s * h

        per_layer_saved = 3 * tokens_h + b * s * i
        embeddings_saved = tokens_h

        return (embeddings_saved + l * per_layer_saved) * db

    def calculate_peak_memory(self) -> int:
        """Total peak memory including FSDP buffers."""
        return (
            self.calculate_param_memory()
            + self.calculate_gradient_memory()
            + self.calculate_optimizer_memory()
            + self.calculate_fsdp_buffer_memory()
            + self.calculate_activation_memory()
        )

    def time_embedding_ms(self) -> float:
        """Embedding lookup time - same as baseline (ms)."""
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        db = self.training.dtype_bytes

        flops = 0
        memory_bytes = 2 * b * s * h * db
        return self.roofline_time_ms(flops, memory_bytes)

    def time_rms_norm_ms(self) -> float:
        """
        Fused RMSNorm time (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        db = self.training.dtype_bytes

        elements = b * s * h
        flops = 8 * elements
        memory_bytes = (elements + h) * db
        return self.roofline_time_ms(flops, memory_bytes)

    def time_attention_ms(self) -> float:
        """
        Flash Attention time (ms).
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

        memory_bytes = 4 * b * s * h * db
        return self.roofline_time_ms(flops, memory_bytes)

    def time_mlp_ms(self) -> float:
        """
        Fused SwiGLU time (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        i = self.model.intermediate_dim
        db = self.training.dtype_bytes

        matmul_flops = 6 * b * s * h * i
        activation_flops = 6 * b * s * i
        flops = matmul_flops + activation_flops

        memory_bytes = (2 * b * s * h + 2 * b * s * i) * db
        return self.roofline_time_ms(flops, memory_bytes)

    def time_lm_head_ms(self) -> float:
        """
        LM head with fused loss (ms).
        """
        return 0.0

    def time_loss_ms(self) -> float:
        """
        Fused linear cross-entropy time (ms).
        """
        b = self.training.batch_size
        s = self.training.seq_len
        h = self.model.hidden_dim
        v = self.model.vocab_size
        db = self.training.dtype_bytes

        flops = 2 * b * s * h * v + 5 * b * s * v
        memory_bytes = (b * s * h + h * v + b * s) * db
        return self.roofline_time_ms(flops, memory_bytes)

    def calculate_allgather_volume(self) -> int:
        """
        FSDP all-gather volume - forward pass (bytes).
        """
        n = self.training.num_gpus
        if n <= 1:
            return 0

        total_param_bf16 = self.calculate_total_params() * self.training.dtype_bytes
        return int(((n - 1) / n) * total_param_bf16)

    def calculate_reducescatter_volume(self) -> int:
        """
        FSDP reduce-scatter volume - backward pass (bytes).
        """
        n = self.training.num_gpus
        if n <= 1:
            return 0

        total_grad_fp32 = self.calculate_total_params() * 4
        return int(((n - 1) / n) * total_grad_fp32)

    def calculate_communication_volume(self) -> int:
        """
        Total FSDP communication volume.
        
        = 2 * all-gather (forward + backward) + reduce-scatter (backward)
        """
        return 2 * self.calculate_allgather_volume() + self.calculate_reducescatter_volume()

    def time_communication_ms(self) -> float:
        """
        FSDP communication time (ms).
        
        time = total_volume / interconnect_bandwidth
        """
        volume = self.calculate_communication_volume()
        bw = self.gpu.interconnect_bandwidth_gbps * 1e9
        return (volume / bw) * 1000

    def overlap_efficiency(self) -> float:
        """
        FSDP overlap efficiency (0.0 to 1.0).
        
        FSDP can overlap:
        - All-gather of next layer with compute of current layer
        - Reduce-scatter of current layer with backward of next layer
        
        Estimate based on your analysis.
        """
        return 0.85

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
