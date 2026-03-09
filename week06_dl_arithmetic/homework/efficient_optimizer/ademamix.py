import math
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end

@torch.compile(fullgraph=True) # you can comment out this line for subtask 1
def ademamix_foreach_fn(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: Tensor,
    lmbda: float,
    bias_correction1: Tensor,
    bias_correction2: Tensor,
    alpha: Tensor,
    eps: float,
    lr: float
):
    if not params:
        return

    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    torch._foreach_lerp_(exp_avgs_slow, grads, 1 - beta3)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    denom = torch._foreach_sqrt(exp_avg_sqs)

    torch._foreach_div_(denom, torch.sqrt(bias_correction2))
    torch._foreach_add_(denom, eps)

    numerator = torch._foreach_div(exp_avgs, bias_correction1)
    scaled_slow = torch._foreach_mul(exp_avgs_slow, alpha)
    torch._foreach_add_(numerator, scaled_slow)

    update = torch._foreach_div(numerator, denom)

    torch._foreach_add_(update, params, alpha=lmbda)
    torch._foreach_add_(params, update, alpha=-lr)

class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0, 
                 beta3_warmup=None, alpha_warmup=None,  eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta3_warmup=beta3_warmup,
                        alpha_warmup=alpha_warmup, weight_decay=weight_decay)
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avgs_slow: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avgs_slow.append(state['exp_avg_slow'])
                exp_avg_sqs.append(state['exp_avg_sq'])

            if not params:
                continue

            step_count = self.state[params[0]]['step']

            bias_correction1 = 1 - beta1 ** step_count
            bias_correction2 = 1 - beta2 ** step_count
            if alpha_warmup is not None:
                alpha = linear_warmup_scheduler(step_count, alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
            else:
                alpha = alpha_final

            if beta3_warmup is not None:
                beta3 = linear_hl_warmup_scheduler(step_count, beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
            else:
                beta3 = beta3_final

            device = params[0].device

            ademamix_foreach_fn(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avgs_slow=exp_avgs_slow,
                exp_avg_sqs=exp_avg_sqs,
                beta1=beta1,
                beta2=beta2,
                beta3=torch.tensor(beta3, device=device, dtype=torch.float32),
                lmbda=lmbda,
                bias_correction1=torch.tensor(bias_correction1, device=device, dtype=torch.float32),
                bias_correction2=torch.tensor(bias_correction2, device=device, dtype=torch.float32),
                alpha=torch.tensor(alpha, device=device, dtype=torch.float32),
                eps=eps,
                lr=lr,
            )
        return loss
