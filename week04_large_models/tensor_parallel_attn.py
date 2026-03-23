from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.integrations.sdpa_attention import sdpa_attention_forward

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaRotaryEmbedding
MODEL_NAME = "unsloth/Llama-3.2-1B"  # for testing (but not grading!), you may want to use Maykeye/TinyLLama-v0
config = LlamaConfig.from_pretrained(MODEL_NAME)
rotary_emb = LlamaRotaryEmbedding(config)

class ComputeWithAllReduce(torch.autograd.Function):
    @staticmethod  # fun fact: torch.distributed.nn has differentiable all_reduce!
    def forward(ctx, tp_shard: nn.Module, input: torch.Tensor, kwargs):
        input = input.detach().requires_grad_(input.requires_grad)
        ctx.save_for_backward(input)
        ctx.kwargs = kwargs
        ctx._tp_shard = tp_shard
        output = tp_shard(input, **kwargs)
        dist.all_reduce(output[0])
        return output
    @staticmethod
    def backward(ctx, *grad_outputs):  # Changed to accept multiple gradients
        grad_output_for_main = grad_outputs[0]
        with torch.enable_grad():
          output = ctx._tp_shard(ctx.saved_tensors[0], **ctx.kwargs)
          torch.autograd.backward(output[0], grad_output_for_main)
        dist.all_reduce(ctx.saved_tensors[0].grad)
        return None, ctx.saved_tensors[0].grad, None


class MyLlamaAttention(nn.Module):
    # please take a reference implementation of Llama attention from Hugging Face transformers:
    # https://github.com/huggingface/transformers/blob/v4.44-release/src/transformers/models/llama/modeling_llama.py#L326-L455
    # You can also directly import transformers.models.llama.modeling_llama.LlamaAttention, as in the reference above.
    # Alternatively, you are welcome to simplify their code or implement your own version.

    # Note: the link above points to an older version of attention with built-in rotary position embeddings (RoPE);
    # If you are using a newer version, please make sure to define extra inputs

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
    
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
# You will likely need to define additional classes below, e.g. a module to perform all-reduce


class AttentionAllReduceModule(MyLlamaAttention):
    def forward(self, input: torch.Tensor, **kwargs):
        return ComputeWithAllReduce.apply(super().forward, input, kwargs)


if __name__ == "__main__":
    dist.init_process_group("gloo")   # use nccl for cuda devices
    torch.manual_seed(1337)           # init weights equally on all ranks
    rank, world_size = dist.get_rank(), dist.get_world_size()

    for active_rank in range(world_size):
      dist.barrier()  # initialize each rank sequentially to save system RAM
      if rank != active_rank: continue

      # we will now implement Tensor Parallelism for the ref_module below:
      ref_module = MyLlamaAttention(config)
      # ^-- you may need to modify this code, e.g. pass parameters or use transformers LlamaAttention (as above)

      # generate reference tensors to test against them later
      input = torch.randn(1, 128, 4096, requires_grad=True)
      extra_inputs = dict()  # <-- OPTIONAL: either design additional inputs here, as in the reference above
      extra_inputs['attention_mask'] = None
      

      input = torch.randn(1, 128, config.hidden_size, requires_grad=True)
      position_embeddings = rotary_emb(input, position_ids=torch.arange(128)[None])
      extra_inputs['position_embeddings'] = position_embeddings

      ref_output = ref_module(input, **extra_inputs)[0]
      ref_output.sum().backward()
      ref_input_grad = input.grad.clone()

      # TP step 1: define a module that computes a portion of attention heads
      tp_config = config
      tp_config.num_attention_heads = config.num_attention_heads // world_size
      tp_config.num_key_value_heads = config.num_key_value_heads // world_size
      tp_module = AttentionAllReduceModule(tp_config)  # create a tensor-parallel version of the Attention module

      with torch.no_grad():
        # copy select weights from the reference attention
        k_start = rank * tp_config.num_key_value_heads * config.head_dim
        k_end = (rank + 1) * tp_config.num_key_value_heads * config.head_dim
        tp_module.k_proj.weight.copy_(ref_module.k_proj.weight[k_start:k_end, :])
        tp_module.v_proj.weight.copy_(ref_module.v_proj.weight[k_start:k_end, :])

        q_start = rank * tp_config.num_attention_heads * config.head_dim
        q_end = (rank + 1) * tp_config.num_attention_heads * config.head_dim
        tp_module.q_proj.weight.copy_(ref_module.q_proj.weight[q_start:q_end, :])

        tp_module.o_proj.weight.copy_(ref_module.o_proj.weight[:, q_start:q_end])

      print(f"Initialized {rank=}", flush=True)
      del ref_module  # free RAM for next rank

    # TEST AREA: you are free to add additional parameters, but your code *must* run the same tests as below
    dist.barrier()  # test 1: forward pass
    tp_input = input.detach().requires_grad_(True)
    tp_output = tp_module(tp_input, **extra_inputs)[0]
    if rank == 0:
        print(f"\nReference outputs ({rank=}):", ref_output.data, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel outputs ({rank=}):", tp_output.data, flush=True)
        assert torch.allclose(tp_output, ref_output, atol=1e-5), f"output mismatch on {rank=}"

    dist.barrier()  # test 2: backward w.r.t. inputs
    assert tp_input.grad is None
    tp_output.sum().backward()
    if rank == 0:
        print(f"\nReference input grad ({rank=}):", ref_input_grad, flush=True)
    for i in range(world_size):
        dist.barrier()
        if i != rank: continue
        print(f"TParallel input grad ({rank=}):", tp_input.grad.data, flush=True)
        assert torch.allclose(tp_input.grad, ref_input_grad, atol=1e-4), f"input_grad mismatch on {rank=}"
