"""
Tests for optimizer step correctness.
"""

import pytest
import torch
import torch.nn as nn

from optimizer.ademamix import AdEMAMix as AdemamixForloop
from efficient_optimizer.ademamix import AdEMAMix as AdemamixForeach

import torch._dynamo as dynamo
dynamo.config.recompile_limit = 8

HIDDEN_DIM = 16
NUM_LAYERS = 3
NUM_STEPS = 100

def _build_model(device: torch.device, dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    layers = [nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True) for _ in range(NUM_LAYERS)]
    return nn.Sequential(*layers).to(device=device, dtype=dtype)


def _assert_models_close(model_a: nn.Module, model_b: nn.Module, step: int, rtol: float=1e-5, atol: float=1e-6) -> None:
    a = dict(model_a.named_parameters())
    b = dict(model_b.named_parameters())
    assert a.keys() == b.keys()

    for name in a.keys():
        pa, pb = a[name].data, b[name].data
        max_diff = (pa - pb).abs().max().item()
        assert torch.allclose(pa, pb, atol=atol, rtol=rtol), (
            f"Param mismatch at step={step}, name={name}, max_diff={max_diff}"
        )


def _apply_random_grads(model_a: nn.Module, model_b: nn.Module) -> None:
    torch.manual_seed(0)
    a = dict(model_a.named_parameters())
    b = dict(model_b.named_parameters())
    for name in a.keys():
        g = torch.randn_like(a[name].data)
        a[name].grad = g
        b[name].grad = g.clone()


class TestCorrectness:
    """Test correctness of efficient AdEMAMix implementation."""
    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.float32, 1e-6, 1e-5),
    ])
    def test_steps_match(self, device, dtype, atol, rtol):
        model_baseline = _build_model(device, dtype)
        model_efficient = _build_model(device, dtype)

        opt_baseline = AdemamixForloop(model_baseline.parameters(), lr=1e-2, weight_decay=0.1, alpha_warmup=51, beta3_warmup=51)
        opt_efficient = AdemamixForeach(model_efficient.parameters(), lr=1e-2, weight_decay=0.1, alpha_warmup=51, beta3_warmup=51)

        _assert_models_close(model_baseline, model_efficient, step=0, atol=atol, rtol=rtol)

        for step in range(1, NUM_STEPS + 1):
            _apply_random_grads(model_baseline, model_efficient)
            opt_baseline.step()
            opt_efficient.step()
            _assert_models_close(model_baseline, model_efficient, step=step, atol=atol, rtol=rtol)

import torch._dynamo as dynamo

class TestCorrectnessZeroBreaks:
    def test_ademamix_foreach_has_zero_graph_breaks(self, device):
        from efficient_optimizer.ademamix import ademamix_foreach_fn

        params = [torch.randn(8, 8, device=device)]
        grads = [torch.randn(8, 8, device=device)]
        exp_avgs = [torch.randn(8, 8, device=device)]
        exp_avgs_slow = [torch.randn(8, 8, device=device)]
        exp_avg_sqs = [torch.rand(8, 8, device=device) + 0.1]

        beta1, beta2 = 0.9, 0.999
        lmbda = 0.1
        eps = 1e-8
        lr = 1e-2

        beta3 = torch.tensor(0.999, device=device, dtype=torch.float32)
        bias_correction1 = torch.tensor(0.9, device=device, dtype=torch.float32)
        bias_correction2 = torch.tensor(0.999, device=device, dtype=torch.float32)
        alpha = torch.tensor(0.5, device=device, dtype=torch.float32)

        explanation = dynamo.explain(ademamix_foreach_fn)(
            params=params,
            grads=grads,
            exp_avgs=exp_avgs,
            exp_avgs_slow=exp_avgs_slow,
            exp_avg_sqs=exp_avg_sqs,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            lmbda=lmbda,
            bias_correction1=bias_correction1,
            bias_correction2=bias_correction2,
            alpha=alpha,
            eps=eps,
            lr=lr,
        )

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}, "
            f"reasons: {explanation.break_reasons}"
        )



class TestNumberOfGeneratedKernels:
    def _run_ademamix_foreach(self, device, dtype):
        torch._inductor.metrics.generated_kernel_count = 0
        from efficient_optimizer.ademamix import AdEMAMix as AdemamixForeach
    
        model = _build_model(device, dtype)
        opt = AdemamixForeach(
            model.parameters(),
            lr=1e-2,
            weight_decay=0.1,
            alpha_warmup=51,
            beta3_warmup=51,
        )

        for step in range(1, NUM_STEPS + 1):
            _apply_random_grads(model, model)
            opt.step()

        return torch._inductor.metrics.generated_kernel_count

    def test_foreach_uses_exactly_one_kernel(self, device):
        dtype = torch.float16
        foreach_kernels = self._run_ademamix_foreach(device, dtype)
        assert foreach_kernels == 1, (
            f"Expected 1 Triton kernel, got {foreach_kernels}"
        )
