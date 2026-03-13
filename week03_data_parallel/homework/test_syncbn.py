import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
import os
import numpy as np
from syncbn import SyncBatchNorm


def setup_process(rank, world_size, fn, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    fn(rank, world_size, *args)
    dist.destroy_process_group()


def run_syncbn(rank, world_size, full_x, batch_size, results):
    hid_dim = full_x.shape[1]
    model = SyncBatchNorm(hid_dim)

    start = rank * batch_size
    end = (rank + 1) * batch_size

    x = full_x[start:end].clone().requires_grad_(True)
    out = model(x)

    global_half = full_x.shape[0] // 2

    local_mask_start = start
    local_mask_end = min(end, global_half)

    if local_mask_start < global_half:
        local_slice_start = 0
        local_slice_end = local_mask_end - start
        loss = out[local_slice_start:local_slice_end].sum()
    else:
        loss = out.sum() * 0

    loss.backward()

    results[rank] = {
        "output": out.detach(),
        "grad": x.grad.detach(),
        "mean": model.running_mean.detach(),
        "var": model.running_var.detach(),
    }



def run_standard_bn(full_x):
    hid_dim = full_x.shape[1]
    model = nn.BatchNorm1d(hid_dim, affine=False)
    x = full_x.clone().requires_grad_(True)

    out = model(x)
    loss = out[: full_x.shape[0] // 2].sum()
    loss.backward()

    return {
        "output": out.detach(),
        "grad": x.grad.detach(),
        "mean": model.running_mean.detach(),
        "var": model.running_var.detach(),
    }


@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("num_workers", [1, 4])
def test_batchnorm(batch_size, hid_dim, num_workers):
    torch.manual_seed(42)
    total = batch_size * num_workers
    full_x = torch.randn(total, hid_dim)

    ctx = mp.get_context("spawn")
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=setup_process,
            args=(rank, num_workers, run_syncbn, (full_x, batch_size, results)),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    outputs = torch.cat([results[i]["output"] for i in range(num_workers)], dim=0)
    grads = torch.cat([results[i]["grad"] for i in range(num_workers)], dim=0)
    means = torch.stack([results[i]["mean"] for i in range(num_workers)])
    vars_ = torch.stack([results[i]["var"] for i in range(num_workers)])

    std = run_standard_bn(full_x)

    torch.testing.assert_close(outputs, std["output"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grads, std["grad"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(means[0], std["mean"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(vars_[0], std["var"], rtol=1e-4, atol=1e-4)
