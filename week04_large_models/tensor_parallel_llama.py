import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers

from tensor_parallel_attn import AttentionAllReduceModule
from tensor_parallel_mlp import AllReduceModule, LlamaMLP

from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaForCausalLM
)

MODEL_NAME = "unsloth/Llama-3.2-1B"
config = LlamaConfig.from_pretrained(MODEL_NAME)
rotary_emb = LlamaRotaryEmbedding(config)

# -----------------------------
# Tensor-parallel block
# -----------------------------
class TPLlamaBlock(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        tp_config = copy.deepcopy(config)
        self.config = tp_config
        tp_config.num_attention_heads = config.num_attention_heads // dist.get_world_size()
        tp_config.num_key_value_heads = config.num_key_value_heads // dist.get_world_size()
        self.self_attn = AttentionAllReduceModule(tp_config)
        self.mlp = AllReduceModule(LlamaMLP(config.hidden_size, config.intermediate_size // dist.get_world_size()))

    def copy_weights_from(self, ref_layer: LlamaDecoderLayer):
        with torch.no_grad():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            # MLP weights
            intermediate_size = ref_layer.mlp.down_proj.in_features
            local_units = intermediate_size // world_size
            unit_slice = slice(local_units * rank, local_units * (rank + 1))
            self.mlp[0].gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight[unit_slice])
            self.mlp[0].up_proj.weight.copy_(ref_layer.mlp.up_proj.weight[unit_slice])
            self.mlp[0].down_proj.weight.copy_(ref_layer.mlp.down_proj.weight[:, unit_slice])
            # Attention weights
            k_start = rank * self.config.num_key_value_heads * self.config.head_dim
            k_end = (rank + 1) * self.config.num_key_value_heads * self.config.head_dim
            self.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight[k_start:k_end, :])
            self.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight[k_start:k_end, :])
            q_start = rank * self.config.num_attention_heads * self.config.head_dim
            q_end = (rank + 1) * self.config.num_attention_heads * self.config.head_dim
            self.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight[q_start:q_end, :])
            self.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight[:, q_start:q_end])

# -----------------------------
# Tensor-parallel LLaMA
# -----------------------------
class MyLlamaModel(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model.layers = nn.ModuleList([TPLlamaBlock(config, i) for i in range(config.num_hidden_layers)])

    def copy_weights_from(self, ref_model: LlamaForCausalLM):
        for tp_layer, ref_layer in zip(self.model.layers, ref_model.model.layers):
            tp_layer.copy_weights_from(ref_layer)

# -----------------------------
# Main distributed routine
# -----------------------------
if __name__ == "__main__":
    # --- 0. Init distributed ---
    print("[BOOT] Process starting", flush=True)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    print(f"[R{rank}] Process group initialized (world_size={world_size})", flush=True)

    torch.manual_seed(1337)

    # --- 1. Load reference model on all ranks (or just rank 0 if CPU memory limited) ---
    print(f"[R{rank}] Loading reference model...", flush=True)
    ref_model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    ref_model.eval()

    # --- 2. Barrier to ensure all ranks finished loading ---
    dist.barrier()
    print(f"[R{rank}] Reference model ready", flush=True)

    # --- 3. Initialize TP model ---
    print(f"[R{rank}] Initializing tensor-parallel model...", flush=True)
    model = MyLlamaModel(config)
    model.copy_weights_from(ref_model)  # shard weights
    model.to(device)
    model.eval()
    print(f"[R{rank}] TP model ready", flush=True)

    # --- 4. Prepare inputs ---
    seq_len = 8
    position_ids = torch.arange(seq_len)[None]
    ref_inputs = torch.randn(1, seq_len, config.hidden_size, requires_grad=True)
    tp_inputs = ref_inputs.detach().clone().to(device).requires_grad_(True)
    position_ids_gpu = position_ids.to(device)

    # --- 5. Compute reference outputs (rank 0 only) ---
    if rank == 0:
        print(f"[R0] Computing reference forward/backward...", flush=True)
        ref_out = ref_model(inputs_embeds=ref_inputs, position_ids=position_ids, attention_mask=None).logits
        ref_out.sum().backward()
        ref_output = ref_out.detach().to(device)
        ref_input_grad = ref_inputs.grad.detach().to(device)
    else:
        ref_output = torch.empty(1, seq_len, config.hidden_size, device=device)
        ref_input_grad = torch.empty_like(ref_output)

    # --- 6. Broadcast reference outputs and grads ---
    print(f"[R{rank}] Broadcasting reference outputs...", flush=True)
    dist.broadcast(ref_output, src=0)
    dist.broadcast(ref_input_grad, src=0)
    dist.barrier()
    print(f"[R{rank}] Broadcast done", flush=True)

    # --- 7. TP forward/backward ---
    tp_out = model(inputs_embeds=tp_inputs, position_ids=position_ids_gpu, attention_mask=None).logits
    tp_out.sum().backward()

    # --- 8. Compare outputs and grads ---
    for i in range(world_size):
        dist.barrier()
        if i == rank:
            print(f"[R{rank}] TP logits:\n", tp_out.data.cpu(), flush=True)
            if rank == 0:
                print(f"[R{rank}] Reference logits:\n", ref_output.cpu(), flush=True)
            assert torch.allclose(tp_out.cpu(), ref_output.cpu(), atol=1e-4), f"Output mismatch on rank {rank}"

    for i in range(world_size):
        dist.barrier()
        if i == rank:
            print(f"[R{rank}] TP input grad:\n", tp_inputs.grad.data.cpu(), flush=True)
            if rank == 0:
                print(f"[R{rank}] Reference input grad:\n", ref_input_grad.cpu(), flush=True)
            assert torch.allclose(tp_inputs.grad.cpu(), ref_input_grad.cpu(), atol=1e-4), f"Grad mismatch on rank {rank}"

    dist.barrier()
    print(f"[R{rank}] ✓ All checks passed!", flush=True)
