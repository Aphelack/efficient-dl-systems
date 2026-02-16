import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class _SyncBatchNormFunction(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps, momentum):
        dims = [0] + list(range(2, input.dim()))
        C = input.size(1)

        count = input.numel() // C
        count_tensor = torch.full(
            (C,), count, dtype=input.dtype, device=input.device
        )

        mean = input.mean(dim=dims)
        mean_x2 = (input * input).mean(dim=dims)

        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

        if distributed:
            vec = torch.cat([
                mean * count_tensor,
                mean_x2 * count_tensor,
                count_tensor
            ])
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            mean_sum = vec[:C]
            x2_sum = vec[C:2 * C]
            total_count = vec[2 * C:]

            mean = mean_sum / total_count
            var = x2_sum / total_count - mean * mean
        else:
            total_count = count_tensor
            var = mean_x2 - mean * mean

        if running_mean is not None:
            running_mean.mul_(1 - momentum).add_(momentum * mean)

        if running_var is not None:
            unbiased_var = var * total_count / (total_count - 1)
            running_var.mul_(1 - momentum).add_(momentum * unbiased_var)

        std = torch.sqrt(var + eps)

        shape = (1, -1) + (1,) * (input.dim() - 2)
        normalized = (input - mean.view(*shape)) / std.view(*shape)

        ctx.save_for_backward(normalized, std, total_count)
        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        normalized, std, total_count = ctx.saved_tensors

        dims = [0] + list(range(2, grad_output.dim()))

        sum_dy = grad_output.sum(dim=dims)
        sum_dy_xhat = (grad_output * normalized).sum(dim=dims)

        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

        if distributed:
            vec = torch.cat([sum_dy, sum_dy_xhat])
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            C = sum_dy.numel()
            sum_dy = vec[:C]
            sum_dy_xhat = vec[C:]

        inv_std = 1.0 / std
        shape = (1, -1) + (1,) * (grad_output.dim() - 2)

        grad_input = inv_std.view(*shape) * (
            grad_output
            - sum_dy.view(*shape) / total_count.view(*shape)
            - normalized * sum_dy_xhat.view(*shape) / total_count.view(*shape)
        )

        return grad_input, None, None, None, None

class SyncBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, input):
        if not self.training:
            shape = (1, -1) + (1,) * (input.dim() - 2)
            return (
                (input - self.running_mean.view(*shape))
                / torch.sqrt(self.running_var.view(*shape) + self.eps)
            ) * self.weight.view(*shape) + self.bias.view(*shape)

        x = _SyncBatchNormFunction.apply(
            input,
            self.running_mean,
            self.running_var,
            self.eps,
            self.momentum,
        )

        shape = (1, -1) + (1,) * (input.dim() - 2)
        return x * self.weight.view(*shape) + self.bias.view(*shape)
