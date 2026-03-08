"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn

@torch.compile
def rmsnorm_forward(x, weight, eps):
    """Zero-Centered RMSNorm forward."""
    # TODO: Replace with fused implementation
    input_dtype = x.dtype
    x = x.float()
    x_squared = x * x
    mean_squared = x_squared.mean(dim=-1, keepdim=True)
    mean_squared_eps = mean_squared + eps
    rsqrt = torch.rsqrt(mean_squared_eps)
    normalized = x * rsqrt
    scale = 1.0 + weight.float()
    output = normalized * scale
    # TODO: Think about additional return parameters
    return output.to(input_dtype), rsqrt

@torch.compile
def rmsnorm_backward(grad_output, x, weight, rsqrt):
    """Zero-Centered RMSNorm backward."""
    x_f32 = x.float()
    x_hat = x_f32 * rsqrt
    scale = 1.0 + weight.float()
    wdy = grad_output * scale
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rsqrt

    reduce_dims = tuple(range(x.ndim - 1))
    dw = (grad_output * x_hat).sum(dim=reduce_dims)
    
    return dx.to(x.dtype), dw.to(weight.dtype)


class RMSNormFunction(torch.autograd.Function):
    """
    Template for memory-efficient and fused Zero-Centered RMSNorm autograd function.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        # TODO: Replace with fused implementation
        output, rsqrt = rmsnorm_forward(x, weight, eps)

        # TODO: Save tensors for backward (make it memory-efficient)
        ctx.save_for_backward(x, weight, rsqrt)  # TODO: Fill this

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement fused backward pass
        # TODO: Make it work with memory-efficient forward
        x, weight, rstd = ctx.saved_tensors

        dx, dw = rmsnorm_backward(grad_output, x, weight, rstd)
        return dx, dw, None


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm: y = x/rms(x) * (1 + weight), weight init to zeros.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
