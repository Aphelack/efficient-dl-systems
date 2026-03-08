"""
gpt-oss style SwiGLU Feed-Forward Network

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def silu(x, alpha=1):
    # TODO: Replace with x * sigmoid(x * alpha)
    return x * tl.sigmoid(x * alpha)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, alpha, limit, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # a = gate, b = up
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    
    # TODO: Add clamping
    a_row = tl.minimum(a_row, limit)
    b_row = tl.maximum(tl.minimum(b_row, limit), -limit)
    
    # TODO: Replace silu(a_row) * b_row with glu * (b_row + 1)
    c_row = silu(a_row, alpha).to(b_row.dtype) * (b_row + 1.)
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, alpha, limit, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    
    # sigmoid requires type float32
    a_orig = tl.load(a_ptr + col_offsets, mask=mask, other=0)
    b_orig = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    a_row = a_orig.to(tl.float32)
    b_row = b_orig.to(tl.float32)

    a_clamped = tl.minimum(a_row, limit)
    b_clamped = tl.maximum(tl.minimum(b_row, limit), -limit)

    # recomputation to save memory
    # TODO: Update backward pass for gpt-oss style implementation (formula will be different!)
    sig_a = tl.sigmoid(a_clamped * alpha)
    silu_a = a_clamped * sig_a
    
    db_row = dc_row * silu_a
    da_row = dc_row * (alpha * silu_a * (1 - sig_a) + sig_a) * (b_clamped + 1.)

    # Mask out gradients if they were outside the clamp limit
    da_row = tl.where(a_row < limit, da_row, 0.0)
    db_row = tl.where((b_row > -limit) & (b_row < limit), db_row, 0.0)

    # Recompute C map 
    c_row = silu_a.to(b_orig.dtype) * (b_clamped + 1.)

    tl.store(a_ptr + col_offsets, da_row.to(a_orig.dtype), mask=mask)
    tl.store(b_ptr + col_offsets, db_row.to(b_orig.dtype), mask=mask)
    
    # HINT: We're already recomputing values here.
    # Could a third store here help avoid saving something else?
    tl.store(dc_ptr + col_offsets, c_row.to(dc_row.dtype), mask=mask)


def swiglu_forward(a, b, alpha, limit):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        alpha,
        limit,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc, alpha, limit):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    # Removed: c = torch.empty_like(dc) (This was wasting exactly the few MBs we needed!)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        alpha,
        limit,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape), dc.view(*ori_shape)


class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    """
    Memory-optimized SwiGLU MLP with selective recomputation.
    """
    
    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit):
        gate = x @ w_gate.T
        up = x @ w_up.T

        # TODO: Replace with fused swiglu_forward kernel
        _, _, activation_out = swiglu_forward(gate, up, alpha, limit)

        out = activation_out @ w_down.T
        
        # Free activation_out BEFORE PyTorch exits the function context to drop peak memory
        del activation_out

        # TODO: Save tensors for backward
        ctx.save_for_backward(x, gate, up, w_gate, w_up, w_down)
        ctx.alpha = alpha
        ctx.limit = limit

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement backward pass
        # ... unpack, recompute whats needed, d_activation ...

        # Gradient through activation
        # ... = swiglu_backward(...)

        # Gradients for w_down, w_gate, w_up, dx
        x, gate, up, w_gate, w_up, w_down = ctx.saved_tensors
        alpha = ctx.alpha
        limit = ctx.limit

        d_activation = grad_output @ w_down

        del w_down
        d_gate, d_up, recomputed_activation = swiglu_backward(gate, up, d_activation, alpha, limit)

        del d_activation
        del gate
        del up
        grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
        recomputed_activation_2d = recomputed_activation.reshape(-1, recomputed_activation.shape[-1])
        
        d_w_down = grad_out_2d.T @ recomputed_activation_2d

        del recomputed_activation
        del recomputed_activation_2d
        del grad_out_2d

        x_2d = x.reshape(-1, x.shape[-1])
        
        d_gate_2d = d_gate.reshape(-1, d_gate.shape[-1])
        d_w_gate = d_gate_2d.T @ x_2d

        d_x_2d = torch.matmul(d_gate_2d, w_gate)

        del d_gate
        del d_gate_2d

        d_up_2d = d_up.reshape(-1, d_up.shape[-1])
        d_w_up = d_up_2d.T @ x_2d

        d_x_2d.addmm_(d_up_2d, w_up)
        
        del d_up
        del d_up_2d
        del x_2d

        d_x = d_x_2d.view_as(x)

        return d_x, d_w_gate, d_w_up, d_w_down, None, None



class SwiGLUFeedForward(nn.Module):
    """
    gpt-oss style SwiGLU.
    
    output = W_down @ ((up + 1) * gate * sigmoid(gate * alpha))
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit
        )
